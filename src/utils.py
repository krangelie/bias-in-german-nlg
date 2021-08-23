def has_context(txt, context_list):
    if any(context in str(txt) for context in context_list):
        return True
    else:
        return False
