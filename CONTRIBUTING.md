# Contributing to Finance RAG System

Thank you for your interest in contributing to the Finance RAG System! We welcome contributions from the community to help improve this project.

## Getting Started

1. **Fork the Repository**
   - Click the "Fork" button in the top-right corner of the repository page
   - Clone your forked repository locally:
     ```bash
     git clone https://github.com/yourusername/FinanceRAGSystem.git
     cd FinanceRAGSystem
     ```

2. **Set Up Development Environment**
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install development dependencies:
     ```bash
     pip install -r requirements-dev.txt
     ```

3. **Create a New Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Making Changes

1. **Code Style**
   - Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
   - Use type hints for all function parameters and return values
   - Keep functions small and focused on a single responsibility

2. **Documentation**
   - Add docstrings to all functions, classes, and modules following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
   - Update the README.md with any new features or changes

3. **Testing**
   - Write tests for new functionality
   - Run tests before submitting a pull request:
     ```bash
     pytest
     ```
   - Ensure all tests pass before submitting a PR

## Submitting a Pull Request

1. **Update Your Fork**
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

2. **Commit Your Changes**
   - Write clear, concise commit messages
   - Reference any related issues in your commit messages

3. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Fill in the PR template with details about your changes

## Code Review Process

1. **Initial Review**
   - A maintainer will review your PR within a few days
   - Be prepared to address any feedback or make requested changes

2. **Addressing Feedback**
   - Make the requested changes
   - Push the updates to your branch
   - The PR will automatically update with your changes

3. **Approval**
   - Once approved, a maintainer will merge your PR
   - Your contribution will be included in the next release!

## Reporting Issues

If you find a bug or have a feature request, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the issue (if applicable)
- Expected vs. actual behavior
- Any relevant error messages or logs

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
