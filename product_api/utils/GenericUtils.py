def convert_to_tuple(input_string):
    # Split the string by commas and strip any extra spaces
    items = input_string.split(',')
    # Trim whitespace and form a tuple
    trimmed_items = [item.strip() for item in items]
    # check the length of trimmed_items
    if len(trimmed_items) == 1:
        return f"('{trimmed_items[0]}')"
    return tuple(trimmed_items)
