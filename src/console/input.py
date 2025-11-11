from .output import colored_text


def select_option(
    options: list[str],
    prompt: str = "Select an option:",
    allow_empty: bool = False,
    case_sensitive: bool = False,
) -> str | None:
    """
    Lets the user select one option from a list.

    Args:
        options: List of available options to choose from.
        prompt: Prompt text displayed before input (default: "Select an option:").
        allow_empty: If True, allows empty input (returns None).
        case_sensitive: If True, case is considered in comparisons.

    Returns:
        The selected option (string) or None for empty input (if allow_empty=True).

    Raises:
        ValueError: If options list is empty.
        KeyboardInterrupt: If user interrupts input (Ctrl+C).
    """
    if not options:
        raise ValueError("Options list cannot be empty")

    # Normalize case if case insensitive
    normalized_options = options
    if not case_sensitive:
        normalized_options = [opt.lower() for opt in options]

    while True:
        # Display options with numbering
        print(prompt)
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            user_input = input("Enter option number or name: ").strip()

            # Handle empty input
            if not user_input:
                if allow_empty:
                    return None
                else:
                    print(colored_text("Please enter a value.", 'red'))
                    continue

            # Check by number
            if user_input.isdigit():
                index = int(user_input) - 1
                if 0 <= index < len(options):
                    return options[index]

            # Check by name
            search_value = user_input if case_sensitive else user_input.lower()
            if search_value in normalized_options:
                # Return original from options (with correct case)
                return options[normalized_options.index(search_value)]

            print(colored_text("Invalid input. Please try again.", 'red'))

        except Exception as _:
            print(colored_text("Input error. Please try again.", 'red'))


def confirm(
    message: str = "Confirm (Y/n)? ",
    yes_options: list[str] | None = None,
    no_options: list[str] | None = None,
    invalid_response_defaults_to_no: bool = True,
) -> bool:
    """
    Asks user for confirmation (yes/no).

    Args:
        message: Message text to display before input.
        yes_options: List of valid 'yes' responses (default: ['y', 'yes']).
        no_options: List of valid 'no' responses (default: ['n', 'no']).
        invalid_response_defaults_to_no: If True, any invalid input counts as 'no'.
                                           If False, prompts for input until valid response.

    Returns:
        True if response matches 'yes' options; False otherwise.
    """
    # Default values
    if yes_options is None:
        yes_options = ["y", "yes"]
    if no_options is None:
        no_options = ["n", "no"]

    # Combine all valid responses
    valid_responses = set(option.lower() for option in yes_options + no_options)

    while True:
        user_input = input(message).strip().lower()

        if user_input in valid_responses:
            return user_input in yes_options

        if invalid_response_defaults_to_no:
            return False


def input_with_default(
    prompt: str,
    default: str,
) -> str:
    """
    Prompts user for input, returning a default value if input is empty.

    Args:
        prompt: Prompt text displayed before input.
        default: Default value to return if input is empty.

    Returns:
        User input string, or default if input is empty.
    """
    user_input = input(f"{prompt} (default: {default}): ").strip()
    return user_input if user_input else default

if __name__ == "__main__":
    print(select_option(["apple", "banana", "cucumber"]))
    print(confirm())
