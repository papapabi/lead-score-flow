#!/usr/bin/env python
"""Lead Score Flow entrypoint and orchestration module.

High-level process performed in this file:
1. Define the shared flow state used across the entire run, including the raw
    candidate data, AI-generated scores, merged ranked candidates, and any
    human feedback supplied during review.
2. Start the flow by loading lead records from `leads.csv`, validating each row
    into the project's candidate model, and storing the resulting candidates in
    state for downstream steps.
3. Score every candidate against the job description with the lead-scoring crew.
    This work is executed asynchronously so multiple candidate evaluations can be
    processed concurrently during each scoring pass.
4. Combine the source candidate data with the returned scores, rank the
    candidates, and present the top results to a human reviewer.
5. Use a router-driven human-in-the-loop checkpoint to decide what happens next:
    quit, re-run scoring with additional reviewer feedback, or continue to email
    generation.
6. If feedback is provided, store that feedback in state and trigger another
    scoring cycle so the crew can reconsider candidates using the new guidance.
7. If the reviewer approves proceeding, generate personalized email responses
    for all candidates in parallel, using the top-ranked candidates to determine
    who advances and who receives a rejection response.
8. Save the generated email text files to an output directory and expose helper
    entrypoints to either run the flow or render a visualization of the flow
    graph.

In short, this module is the control layer for the lead-scoring workflow: it
loads inputs, coordinates AI crews, manages review feedback, branches the flow,
and persists the final communication artifacts.
"""

import asyncio
from typing import List

# CrewAI Flow primitives:
#   Flow       – base class that wires decorated methods into a directed execution graph
#   @start()   – marks the first method to run when the flow is kicked off
#   @listen()  – subscribes a method to one or more upstream events/methods
#   @router()  – like @listen, but the return value determines which downstream branch runs next
#   or_()      – logical OR helper: the decorated method fires when ANY of the listed events emit
from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel

# The full job description string used as context when scoring each candidate
from lead_score_flow.constants import JOB_DESCRIPTION

# LeadResponseCrew – writes personalised accept/reject emails for each candidate
from lead_score_flow.crews.lead_response_crew.lead_response_crew import LeadResponseCrew

# LeadScoreCrew – evaluates a single candidate against the job description
# and returns a structured CandidateScore (id, score 0-100, reason)
from lead_score_flow.crews.lead_score_crew.lead_score_crew import LeadScoreCrew

# Pydantic data models shared across the project:
#   Candidate       – raw CSV row (id, name, email, bio, skills)
#   CandidateScore  – AI output (id, score, reason)
#   ScoredCandidate – merged view of both, used for ranking and email generation
from lead_score_flow.types import Candidate, CandidateScore, ScoredCandidate

# Helper that zips Candidate and CandidateScore lists together by id,
# producing a flat list of ScoredCandidate objects
from lead_score_flow.utils.candidateUtils import combine_candidates_with_scores


# ---------------------------------------------------------------------------
# Flow state
# ---------------------------------------------------------------------------
# LeadScoreState is the shared, mutable "memory" of the entire flow run.
# Every step reads from and writes to this object, so all stages stay in sync
# without needing to pass data explicitly between methods.
class LeadScoreState(BaseModel):
    # All candidates loaded from leads.csv at startup
    candidates: List[Candidate] = []

    # Raw AI scoring results – one CandidateScore per candidate, populated
    # incrementally as async scoring tasks complete
    candidate_score: List[CandidateScore] = []

    # Merged & sorted list combining each Candidate with their AI score.
    # After human_in_the_loop runs, index 0 is the highest-scoring candidate.
    hydrated_candidates: List[ScoredCandidate] = []

    # Optional free-text guidance the human reviewer can provide to steer a
    # second scoring pass (e.g. "prioritise TypeScript experience").
    # Empty string on the first pass – crews ignore it when blank.
    scored_leads_feedback: str = ""


# ---------------------------------------------------------------------------
# Flow definition
# ---------------------------------------------------------------------------
# LeadScoreFlow[LeadScoreState] binds the state type so that self.state is
# always typed as LeadScoreState throughout all methods.
class LeadScoreFlow(Flow[LeadScoreState]):
    # Tell CrewAI which class to instantiate as the initial state object
    # This is not in docs but it makes state initialization more explicit and version-safe.
    # Doing this in later versions causes an error with Flow state having an 'id' field.
    #initial_state = LeadScoreState

    # -----------------------------------------------------------------------
    # Step 1 – Load leads
    # -----------------------------------------------------------------------
    # @start() designates this as the entry point of the flow graph.
    # It runs first and emits an implicit event named after the method
    # ("load_leads") when it returns, which downstream listeners can subscribe to.
    @start()
    def load_leads(self):
        import csv
        from pathlib import Path

        # Resolve leads.csv relative to this file so the path works regardless
        # of the working directory the script is launched from
        current_dir = Path(__file__).parent
        csv_file = current_dir / "leads.csv"

        candidates = []
        with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
            # DictReader maps each row to a dict keyed by the header columns,
            # which maps directly onto the Candidate model field names
            reader = csv.DictReader(file)
            for row in reader:
                print("Row:", row)
                # Unpack the row dict into the Candidate Pydantic model.
                # Pydantic validates the fields and raises early if data is malformed.
                candidate = Candidate(**row)
                candidates.append(candidate)

        # Persist the loaded candidates into shared flow state so every
        # downstream step can access them without re-reading the file
        self.state.candidates = candidates

    # -----------------------------------------------------------------------
    # Step 2 – Score leads (async, concurrent)
    # -----------------------------------------------------------------------
    # @listen(or_(load_leads, "scored_leads_feedback")) means this method fires
    # in two distinct situations:
    #   a) After load_leads completes on the very first run
    #   b) After human_in_the_loop routes back to "scored_leads_feedback"
    #      (i.e. the human asked for a re-score with new feedback)
    # Using or_() lets a single method serve as the handler for both triggers.
    @listen(or_(load_leads, "scored_leads_feedback"))
    async def score_leads(self):
        print("Scoring leads")
        tasks = []

        # Inner coroutine that scores ONE candidate.
        # Defined here, inside, so it closes over `self` and can update shared state directly.
        async def score_single_candidate(candidate: Candidate):
            # kickoff_async launches the LeadScoreCrew as a non-blocking coroutine.
            # Inputs are injected into the crew's agent/task prompt templates.
            # additional_instructions is blank on the first pass; on a re-score
            # it carries the human reviewer's feedback from state.
            result = await (
                LeadScoreCrew()
                .crew()
                .kickoff_async(
                    inputs={
                        "candidate_id": candidate.id,
                        "name": candidate.name,
                        "bio": candidate.bio,
                        "job_description": JOB_DESCRIPTION,
                        "additional_instructions": self.state.scored_leads_feedback,
                    }
                )
            )

            # result.pydantic is the crew's structured output parsed into a
            # CandidateScore model (id, score 0-100, reason).
            # Appending here is safe because asyncio is single-threaded –
            # list.append() is not interrupted between coroutine yields.
            self.state.candidate_score.append(result.pydantic)

        # Build a coroutine task for every candidate.
        # asyncio.create_task() schedules each coroutine on the event loop
        # immediately, so they can start running concurrently right away.
        for candidate in self.state.candidates:
            print("Scoring candidate:", candidate.name)
            task = asyncio.create_task(score_single_candidate(candidate))
            tasks.append(task)

        # asyncio.gather() suspends here until ALL scoring tasks have finished,
        # then returns a list of their results (all None in this case because
        # score_single_candidate doesn't explicitly return a value).
        # The real results were already appended to self.state.candidate_score above.
        candidate_scores = await asyncio.gather(*tasks)
        print("Finished scoring leads: ", len(candidate_scores))

    # -----------------------------------------------------------------------
    # Step 3 – Human-in-the-loop gate (router)
    # -----------------------------------------------------------------------
    # @router(score_leads) runs immediately after score_leads completes.
    # Unlike @listen, a router's return value is treated as a named event
    # string that determines which branch of the flow executes next.
    # This is the mechanism that implements the human approval/feedback loop.
    @router(score_leads)
    def human_in_the_loop(self):
        print("Finding the top 3 candidates for human to review")

        # Merge the raw Candidate list with the AI-generated CandidateScore list
        # into a single ScoredCandidate list keyed by candidate id
        self.state.hydrated_candidates = combine_candidates_with_scores(
            self.state.candidates, self.state.candidate_score
        )

        # Sort descending by score so the best candidates bubble to the top.
        # The sorted list is stored back to state so write_and_save_emails
        # can rely on index order (index 0 = best candidate).
        sorted_candidates = sorted(
            self.state.hydrated_candidates, key=lambda c: c.score, reverse=True
        )
        self.state.hydrated_candidates = sorted_candidates

        # Only the top 3 are shown to the human for review
        top_candidates = sorted_candidates[:3]

        print("Here are the top 3 candidates:")
        for candidate in top_candidates:
            print(
                f"ID: {candidate.id}, Name: {candidate.name}, Score: {candidate.score}, Reason: {candidate.reason}"
            )

        # ---- Human decision point ----
        # The flow pauses here and waits for terminal input.
        # The return value routes execution to the appropriate next step.
        print("\nPlease choose an option:")
        print("1. Quit")
        print("2. Redo lead scoring with additional feedback")
        print("3. Proceed with writing emails to all leads")

        choice = input("Enter the number of your choice: ")

        if choice == "1":
            # Hard exit – terminates the Python process entirely
            print("Exiting the program.")
            exit()
        elif choice == "2":
            # Capture free-text guidance from the reviewer, save it to state,
            # then return the string "scored_leads_feedback".
            # score_leads listens on or_(load_leads, "scored_leads_feedback"),
            # so returning this string re-triggers scoring with the new context.
            # Note: candidate_score is NOT cleared here, so scores from the
            # previous pass are still in state when the new pass appends to it.
            feedback = input(
                "\nPlease provide additional feedback on what you're looking for in candidates:\n"
            )
            self.state.scored_leads_feedback = feedback
            print("\nRe-running lead scoring with your feedback...")
            return "scored_leads_feedback"
        elif choice == "3":
            # Return the string "generate_emails" – write_and_save_emails
            # listens for this exact event name and will run next.
            print("\nProceeding to write emails to all leads.")
            return "generate_emails"
        else:
            # Any other input loops back to the same router step by returning
            # its own method name, giving the user another chance to choose.
            print("\nInvalid choice. Please try again.")
            return "human_in_the_loop"

    # -----------------------------------------------------------------------
    # Step 4 – Write and save emails (async, concurrent)
    # -----------------------------------------------------------------------
    # @listen("generate_emails") subscribes this method to the named event
    # emitted by the router when the human chooses option 3.
    # String-based event names decouple the router from this method –
    # the router doesn't need to import or reference write_and_save_emails directly.
    @listen("generate_emails")
    async def write_and_save_emails(self):
        import re
        from pathlib import Path

        print("Writing and saving emails for all leads.")

        # Build a set of ids for O(1) membership checks below.
        # hydrated_candidates is already sorted best-first, so [:3] are the top scorers.
        # Top candidates receive an "advance to next round" email;
        # others receive a polite rejection email (decided inside LeadResponseCrew).
        top_candidate_ids = {
            candidate.id for candidate in self.state.hydrated_candidates[:3]
        }

        tasks = []

        # Ensure the output directory exists before any async tasks try to write into it.
        # mkdir(parents=True, exist_ok=True) is a no-op if the folder already exists.
        output_dir = Path(__file__).parent / "email_responses"
        print("output_dir:", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Inner coroutine that generates and saves ONE candidate's email
        async def write_email(candidate):
            # This boolean flag is passed to the crew so it can tailor the tone:
            # True  → invitation to proceed / next interview steps
            # False → respectful rejection with encouragement
            proceed_with_candidate = candidate.id in top_candidate_ids

            # kickoff_async runs the LeadResponseCrew non-blocking so all
            # emails can be generated in parallel rather than sequentially
            result = await (
                LeadResponseCrew()
                .crew()
                .kickoff_async(
                    inputs={
                        "candidate_id": candidate.id,
                        "name": candidate.name,
                        "bio": candidate.bio,
                        "proceed_with_candidate": proceed_with_candidate,
                    }
                )
            )

            # Strip characters that are illegal or problematic in file names
            # (e.g. slashes, colons, emoji) before using the name as a filename
            safe_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", candidate.name)
            filename = f"{safe_name}.txt"
            print("Filename:", filename)

            # result.raw is the plain-text string output from the crew's final task.
            # Write it directly to a .txt file named after the candidate.
            file_path = output_dir / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result.raw)

            return f"Email saved for {candidate.name} as {filename}"

        # Schedule a write_email coroutine for every scored candidate
        # (not just the top 3 – every lead receives either an advance or rejection email)
        for candidate in self.state.hydrated_candidates:
            task = asyncio.create_task(write_email(candidate))
            tasks.append(task)

        # Wait for all email tasks to finish concurrently, then collect their
        # confirmation messages to print a final summary
        email_results = await asyncio.gather(*tasks)

        print("\nAll emails have been written and saved to 'email_responses' folder.")
        for message in email_results:
            print(message)


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def kickoff():
    """
    Instantiate and synchronously run the full LeadScoreFlow.
    Called by the CLI entrypoint defined in pyproject.toml and by __main__.
    """
    lead_score_flow = LeadScoreFlow(tracing=True)
    lead_score_flow.kickoff()


def plot():
    """
    Render an interactive HTML visualisation of the flow graph and open it
    in the browser. Useful for understanding the step dependencies at a glance.
    """
    lead_score_flow = LeadScoreFlow()
    lead_score_flow.plot()


# Allow the module to be run directly with `python main.py` in addition to
# being invoked through the package CLI entrypoint
# But this is only optional. The intended way is to run `crewai flow kickoff` in the project root.
if __name__ == "__main__":
    kickoff()
