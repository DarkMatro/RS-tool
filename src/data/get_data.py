def get_parent(parent, class_name: str):
    if parent.__class__.__name__ == class_name:
        return parent
    return get_parent(parent.parent, class_name)
