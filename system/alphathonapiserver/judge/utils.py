def write_file(path: str, content) -> None:
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode=mode) as writer:
        writer.write(content)
