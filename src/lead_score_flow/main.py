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
5. Use an AMP-compatible human-input checkpoint (email/web review in AMP,
    interactive approval locally) to decide what happens next: quit, revise
    scoring guidance, or approve email generation.
6. If revision feedback is provided, store it in state and trigger another
    scoring cycle so the crew can re-evaluate candidates using the new guidance.
7. If the reviewer approves, generate personalized email responses
    for all candidates in parallel, using the top-ranked candidates to determine
    who advances and who receives a rejection response.
8. Save the generated email text files to an output directory and expose helper
    entrypoints to either run the flow or render a visualization of the flow
    graph.

In short, this module is the control layer for the lead-scoring workflow: it
loads inputs, coordinates AI crews, manages review feedback, branches the flow,
and persists the final communication artifacts.
"""

import argparse
import asyncio
import importlib
import os
import sys
from typing import Any, Callable, Dict, List

# CrewAI Flow primitives:
#   Flow       – base class that wires decorated methods into a directed execution graph
#   @start()   – marks the first method to run when the flow is kicked off
#   @listen()  – subscribes a method to one or more upstream events/methods
#   or_()      – logical OR helper: the decorated method fires when ANY of the listed events emit
#   @human_input() – creates a deploy-safe human-in-the-loop checkpoint with emitted labels
from crewai.flow.flow import Flow, listen, or_, start

# CrewAI renamed this decorator across versions. Keep a compatibility fallback
# so the flow works both locally and when deployed to AMP.
try:
    human_input = importlib.import_module(
        "crewai.flow.human_input"
    ).human_input
except (ModuleNotFoundError, AttributeError):  # pragma: no cover - version fallback
    human_input = importlib.import_module(
        "crewai.flow.human_feedback"
    ).human_feedback

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

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
    # After `request_human_review` runs, index 0 is the highest-scoring candidate.
    hydrated_candidates: List[ScoredCandidate] = []

    # Optional free-text guidance the human reviewer can provide to steer a
    # second scoring pass (e.g. "prioritise TypeScript experience").
    # Empty string on the first pass – crews ignore it when blank.
    revision_feedback: str = ""


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
    # @listen(or_(load_leads, "revise_requested")) means this method fires
    # in two distinct situations:
    #   a) After load_leads completes on the very first run
    #   b) After `handle_revise` emits "revise_requested"
    #      (i.e. the human asked for a scoring revision with new feedback)
    # Using or_() lets a single method serve as the handler for both triggers.
    @listen(or_(load_leads, "revise_requested"))
    async def score_leads(self):
        print("Scoring leads")
        tasks = []

        # Clear previous pass scores before re-running. This keeps each scoring
        # cycle deterministic and avoids mixing stale scores with new ones.
        self.state.candidate_score = []

        # Inner coroutine that scores ONE candidate.
        # Defined here, inside, so it closes over `self` and can update shared state directly.
        async def score_single_candidate(candidate: Candidate) -> None:
            # kickoff_async launches the LeadScoreCrew as a non-blocking coroutine.
            # Inputs are injected into the crew's agent/task prompt templates.
            # additional_instructions is blank on the first pass; on a revision
            # pass it carries the human reviewer's feedback from state.
            result = await (
                LeadScoreCrew()
                .crew()
                .kickoff_async(
                    inputs={
                        "candidate_id": candidate.id,
                        "name": candidate.name,
                        "bio": candidate.bio,
                        "job_description": JOB_DESCRIPTION,
                        "additional_instructions": self.state.revision_feedback,
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
    # Step 3 – Human-in-the-loop review checkpoint (AMP-compatible)
    # -----------------------------------------------------------------------
    # `request_human_review` is triggered after scoring completes.
    # `@human_input` emits one of: approve, revise, quit.
    # This works for local runs and AMP deployments.

    def _hydrate_and_rank_candidates(self) -> None:
        """Merge source candidates with scores and sort highest-first."""
        print("Finding the top 3 candidates for human review")

        # Merge the raw Candidate list with the AI-generated CandidateScore list
        # into a single ScoredCandidate list keyed by candidate id
        hydrated_candidates = combine_candidates_with_scores(
            self.state.candidates, self.state.candidate_score
        )

        # Sort descending by score so the best candidates bubble to the top.
        # The sorted list is stored back to state so `generate_and_save_candidate_emails`
        # can rely on index order (index 0 = best candidate).
        sorted_candidates = sorted(
            hydrated_candidates, key=lambda candidate: candidate.score, reverse=True
        )
        self.state.hydrated_candidates = sorted_candidates

    def _extract_last_human_input_text(self) -> str:
        """Read feedback text defensively across CrewAI versions."""
        last_input = getattr(self, "last_human_feedback", "")
        if isinstance(last_input, str):
            return last_input.strip()

        for attr_name in ("feedback", "raw", "message", "input", "response"):
            value = getattr(last_input, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return str(last_input).strip()

    @listen(score_leads)
    @human_input(
        message=(
            "Review the top candidates and choose one action: "
            "approve, revise, or quit. "
            "If you choose revise, include specific scoring guidance."
        ),
        emit=["approve", "revise", "quit"],
        llm="openai/gpt-4o-mini",
    )
    def request_human_review(self) -> str:
        """Create a human review prompt and emit approve/revise/quit."""
        self._hydrate_and_rank_candidates()

        # Only the top 3 are shown to the human for review
        top_candidates = self.state.hydrated_candidates[:3]

        lines = ["Top 3 candidates:"]
        for candidate in top_candidates:
            lines.append(
                (
                    f"- ID: {candidate.id}, Name: {candidate.name}, "
                    f"Score: {candidate.score}, Reason: {candidate.reason}"
                )
            )
        lines.append("")
        lines.append("Reply with one action: approve | revise | quit")
        lines.append("If revise, include specific scoring guidance.")
        return "\n".join(lines)

    @listen("revise")
    def handle_revise(self) -> str:
        """Store revision guidance and trigger a new scoring pass."""
        feedback_text = self._extract_last_human_input_text()
        normalized_feedback = feedback_text

        # Some responses may arrive as "revise: <feedback>".
        # Normalize by stripping the action prefix when present.
        if ":" in feedback_text:
            action, payload = feedback_text.split(":", 1)
            if action.strip().lower() == "revise" and payload.strip():
                normalized_feedback = payload.strip()

        self.state.revision_feedback = normalized_feedback
        print("Re-running lead scoring with revised guidance.")
        return "revise_requested"

    @listen("quit")
    def handle_quit(self) -> None:
        """Stop after review when the human chooses to quit."""
        print("Human reviewer chose quit. Email generation has been skipped.")

    # -----------------------------------------------------------------------
    # Step 4 – Write and save emails (async, concurrent)
    # -----------------------------------------------------------------------
    # @listen("approve") subscribes this method to the approve event emitted
    # by `request_human_review`.
    @listen("approve")
    async def generate_and_save_candidate_emails(self) -> None:
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
        async def write_email(candidate: ScoredCandidate) -> str:
            # This boolean flag is passed to the crew so it can tailor the tone:
            # True  → approved candidate invitation / next interview steps
            # False → respectful rejection with encouragement
            candidate_approved = candidate.id in top_candidate_ids

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
                        "candidate_approved": candidate_approved,
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
    lead_score_flow.plot("flow_visualization.html", show=True)


def _resolve_crewai_test_runtime() -> tuple[int, str]:
    """Resolve runtime parameters for the `crewai test` command."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-n",
        "--n_iterations",
        type=int,
        default=int(os.getenv("CREWAI_TEST_N_ITERATIONS", "2")),
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=os.getenv("CREWAI_TEST_MODEL", "gpt-4o-mini"),
    )
    known_args, unknown_args = parser.parse_known_args(sys.argv[1:])

    # Backward-compatible positional fallback: "<n_iterations> <model>"
    if len(unknown_args) >= 2 and unknown_args[0].isdigit():
        return int(unknown_args[0]), unknown_args[1]

    return known_args.n_iterations, known_args.model


def _run_crew_test_summary(
    crew_builder: Callable[[], Any],
    *,
    inputs: Dict[str, Any],
    n_iterations: int,
    model_name: str,
    crew_label: str,
) -> Dict[str, Any]:
    """Run one crew test pass and return structured metrics for aggregation."""
    from crewai.utilities.evaluators.crew_evaluator_handler import CrewEvaluator
    from crewai.utilities.llm_utils import create_llm

    llm_instance = create_llm(model_name)
    if not llm_instance:
        raise ValueError(f"Failed to create evaluator LLM for model '{model_name}'")

    test_crew = crew_builder().copy()
    evaluator = CrewEvaluator(test_crew, llm_instance)

    for iteration in range(1, n_iterations + 1):
        evaluator.set_iteration(iteration)
        test_crew.kickoff(inputs=inputs)

    return {
        "crew_label": crew_label,
        "crew": test_crew,
        "evaluator": evaluator,
    }


def _print_aggregated_test_results(
    results: List[Dict[str, Any]], n_iterations: int
) -> None:
    """Print one consolidated table for all tested crews."""
    table = Table(
        title="Aggregated CrewAI Test Scores\n(1-10 Higher is better)",
    )
    table.add_column("Crew", style="magenta")
    table.add_column("Tasks/Crew/Agents", style="cyan")
    for run in range(1, n_iterations + 1):
        table.add_column(f"Run {run}", justify="center")
    table.add_column("Avg. Total", justify="center")
    table.add_column("Agents", style="green")

    for result in results:
        crew_label = result["crew_label"]
        crew = result["crew"]
        evaluator = result["evaluator"]

        task_averages = [
            sum(scores) / len(scores)
            for scores in zip(*evaluator.tasks_scores.values(), strict=False)
        ]

        for task_index, task in enumerate(crew.tasks):
            task_scores = [
                evaluator.tasks_scores[run][task_index]
                for run in range(1, n_iterations + 1)
            ]
            avg_score = task_averages[task_index]
            agents = list(task.processed_by_agents)

            table.add_row(
                crew_label,
                f"Task {task_index + 1}",
                *[f"{score:.1f}" for score in task_scores],
                f"{avg_score:.1f}",
                f"- {agents[0]}" if agents else "",
            )

            for agent in agents[1:]:
                table.add_row("", "", *["" for _ in range(n_iterations)], "", f"- {agent}")

        crew_scores = [
            sum(evaluator.tasks_scores[run]) / len(evaluator.tasks_scores[run])
            for run in range(1, n_iterations + 1)
        ]
        crew_average = sum(crew_scores) / len(crew_scores)
        table.add_row(
            crew_label,
            "Crew",
            *[f"{score:.2f}" for score in crew_scores],
            f"{crew_average:.1f}",
            "",
        )

        run_exec_times = [
            int(sum(evaluator.run_execution_times.get(run, [])))
            for run in range(1, n_iterations + 1)
        ]
        execution_time_avg = int(sum(run_exec_times) / len(run_exec_times))
        table.add_row(
            crew_label,
            "Execution Time (s)",
            *map(str, run_exec_times),
            f"{execution_time_avg}",
            "",
        )

    overall_scores = []
    overall_exec_times = []
    for run in range(1, n_iterations + 1):
        run_task_scores = []
        run_exec_total = 0
        for result in results:
            evaluator = result["evaluator"]
            run_task_scores.extend(evaluator.tasks_scores.get(run, []))
            run_exec_total += int(sum(evaluator.run_execution_times.get(run, [])))

        if not run_task_scores:
            continue

        overall_scores.append(sum(run_task_scores) / len(run_task_scores))
        overall_exec_times.append(run_exec_total)

    if overall_scores and overall_exec_times:
        overall_avg = sum(overall_scores) / len(overall_scores)
        overall_exec_avg = int(sum(overall_exec_times) / len(overall_exec_times))
        table.add_row(
            "ALL CREWS",
            "Overall Crew Score",
            *[f"{score:.2f}" for score in overall_scores],
            f"{overall_avg:.1f}",
            "",
        )
        table.add_row(
            "ALL CREWS",
            "Overall Execution Time (s)",
            *map(str, overall_exec_times),
            f"{overall_exec_avg}",
            "",
        )

    console = Console()
    console.print("\n")
    console.print(table)


def test() -> None:
    """Run CrewAI tests for both crews with shared runtime settings."""
    n_iterations, model_name = _resolve_crewai_test_runtime()

    print(
        "Running CrewAI tests for both crews with "
        f"n_iterations={n_iterations}, model={model_name}"
    )

    score_summary = _run_crew_test_summary(
        lambda: LeadScoreCrew().crew(),
        inputs={
            "candidate_id": "24",
            "name": "Ethan Reed",
            "bio": (
                "Self-taught web developer with full-stack experience and AI-driven "
                "solutions. Built projects with React, Next.js, Vercel AI SDK, and "
                "REST APIs."
            ),
            "job_description": JOB_DESCRIPTION,
            "additional_instructions": (
                "Prioritize practical React/Next.js experience and AI integration."
            ),
        },
        n_iterations=n_iterations,
        model_name=model_name,
        crew_label="LeadScoreCrew",
    )

    response_summary = _run_crew_test_summary(
        lambda: LeadResponseCrew().crew(),
        inputs={
            "candidate_id": "24",
            "name": "Ethan Reed",
            "bio": (
                "Self-taught web developer with full-stack experience and AI-driven "
                "solutions."
            ),
            "candidate_approved": True,
        },
        n_iterations=n_iterations,
        model_name=model_name,
        crew_label="LeadResponseCrew",
    )

    _print_aggregated_test_results(
        results=[score_summary, response_summary],
        n_iterations=n_iterations,
    )


# Allow the module to be run directly with `python main.py` in addition to
# being invoked through the package CLI entrypoint
# But this line is only optional. The intended way is to run `crewai flow kickoff` in the project root.
if __name__ == "__main__":
    kickoff()
