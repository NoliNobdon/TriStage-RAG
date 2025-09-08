# Best Practices for Using Claude Code

## Writing Effective Prompts

### Clarity and Specificity
- **Be explicit**: State exactly what you want to accomplish
- **Provide context**: Include relevant background information about your project
- **Set clear requirements**: Specify constraints, technologies, and expected outcomes
- **Use examples**: Show sample inputs/outputs when helpful

### Structure Your Requests
- **Single task per prompt**: Focus on one objective at a time
- **Break down complex tasks**: Use multi-step approaches for larger projects
- **Prioritize tasks**: Indicate what's most important if multiple items need attention
- **Reference specific files**: Use exact paths when mentioning files

### Technical Context
- **Specify your stack**: Mention frameworks, languages, and tools you're using
- **Include error messages**: Paste full error outputs when debugging
- **Show current code**: Provide relevant snippets for context
- **State your environment**: Mention OS, runtime versions, and special configurations

## Avoiding Common Mistakes

### Communication Issues
- **Don't assume**: I don't know your project details unless you tell me
- **Avoid ambiguity**: Clear requests get better results than vague ones
- **Don't omit constraints**: Performance, security, and compatibility requirements matter
- **Never assume I remember**: Each session starts fresh; provide context

### Technical Pitfalls
- **Don't skip testing**: Always run tests and build processes after changes
- **Avoid secrets**: Never include API keys, passwords, or sensitive data
- **Don't ignore errors**: Address all type errors and linting issues
- **Avoid breaking changes**: Test modifications thoroughly

### Workflow Mistakes
- **Don't rush**: Take time to provide complete information
- **Avoid context switching**: Complete one task before starting another
- **Don't skip verification**: Always confirm changes work as expected
- **Avoid over-reliance**: Review my code and understand what it does

## Maximizing Coding Ability

### Leverage My Strengths
- **Code analysis**: I can quickly understand and explain existing code
- **Pattern recognition**: I identify best practices and common patterns
- **Multi-language support**: I work across various programming languages
- **Refactoring expertise**: I can improve code structure and maintainability

### Optimize Our Collaboration
- **Use TodoWrite**: For complex tasks, let me track progress with todo lists
- **Batch operations**: I can handle multiple file operations efficiently
- **Iterative approach**: Build and validate incrementally
- **Ask questions**: I'll clarify uncertainties rather than guess

### Technical Best Practices
- **Follow existing conventions**: I adapt to your project's coding style
- **Security first**: I prioritize secure coding practices
- **Performance awareness**: I consider efficiency and scalability
- **Testing integration**: I ensure code is testable and verify functionality

## Advanced Techniques

### Complex Task Management
- **Break into phases**: Use planning for large projects
- **Set milestones**: Define clear checkpoints for progress
- **Parallel processing**: I can handle multiple operations simultaneously
- **Error handling**: Build robust error handling into your requests

### Debugging and Troubleshooting
- **Provide full context**: Include logs, errors, and relevant code
- **Describe expected vs actual**: Clearly state what should happen vs what does
- **Include environment details**: OS, versions, dependencies matter
- **Show reproduction steps**: List exact steps to reproduce issues

### Code Quality
- **Request reviews**: Ask me to review code for improvements
- **Style consistency**: I'll follow your project's existing patterns
- **Documentation**: I can add comments and documentation when requested
- **Testing**: I'll write and run tests to verify functionality

## Example Effective Prompts

### Good: "Fix the authentication bug in src/auth/login.py where users can't log in with valid credentials"
### Bad: "Fix login"

### Good: "Add error handling to the API client in src/client/api.js. Handle network timeouts and server errors gracefully"
### Bad: "Make the client better"

### Good: "Refactor the User class in models/user.py to use dependency injection instead of direct database connections"
### Bad: "Improve the User class"

## Getting Help

### Built-in Assistance
- Use `/help` for command-line interface help
- Check `CLAUDE.md` for project-specific guidelines
- Ask for clarification if you're unsure about anything

### Feedback Loop
- Report issues at https://github.com/anthropics/claude-code/issues
- Provide specific examples when something doesn't work as expected
- Suggest improvements based on your experience

## Remember

- I'm here to help you be more productive
- Clear communication leads to better results
- Quality takes precedence over speed
- Security and best practices are always priorities
- Don't hesitate to ask for clarification or more detail