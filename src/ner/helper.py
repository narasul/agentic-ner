def extract_tag(llm_output: str, tag: str) -> str:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    tag_len = len(open_tag)

    start_position = llm_output.find(open_tag)
    if start_position == -1:
        return ""

    end_position = llm_output.find(close_tag)

    try:
        return llm_output[start_position + tag_len : end_position]
    except Exception as err:
        print(f"Error while parsing llm output: {str(err)}")
        return ""
