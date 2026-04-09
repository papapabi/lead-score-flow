# API/Backend Development Guidelines

## Programming Language: Python

**Python Best Practices:**
- Follow PEP 8 style guidelines strictly
- Use type hints for all function parameters and return values
- Prefer f-strings for string formatting over older methods
- Use descriptive variable and function names
- Implement proper error handling with specific exception types
- Use virtual environments for dependency management

## Framework: CrewAI
**CrewAI Best Practices:**
- Refer to AGENTS.md for CrewAI-specific guidelines and patterns


## Code Style: Clean Code

**Clean Code Principles:**
- Write self-documenting code with meaningful names
- Keep functions small and focused on a single responsibility
- Avoid deep nesting and complex conditional statements
- Use consistent formatting and indentation
- Write code that tells a story and is easy to understand
- Refactor ruthlessly to eliminate code smells

## Testing: PyTest

**Testing Guidelines:**
- Write comprehensive unit tests for all business logic
- Follow the AAA pattern: Arrange, Act, Assert
- Maintain good test coverage (aim for 80%+ for critical paths)
- Write descriptive test names that explain the expected behavior
- Use test doubles (mocks, stubs, spies) appropriately
- Implement integration tests for API endpoints and user flows
- Keep tests fast, isolated, and deterministic

## Project-Specific Guidelines

These instructions are for CrewAI projects. For CrewAI-specific guidance, refer to the root-level `AGENTS.md` file.
- Follow the Python style guide (https://google.github.io/styleguide/pyguide.html) for coding style and documentation but keep line length and other coding guidelines compatible with the `ruff` linter and formatter.

## AI Code Generation Preferences

When generating code, please:

- Generate complete, working code examples with proper imports
- Include inline comments for complex logic and business rules
- Follow the established patterns and conventions in this project
- Suggest improvements and alternative approaches when relevant
- Consider performance, security, and maintainability
- Include error handling and edge case considerations
- Generate appropriate unit tests when creating new functions
- Follow accessibility best practices for UI components
- Use semantic HTML and proper ARIA attributes when applicable
