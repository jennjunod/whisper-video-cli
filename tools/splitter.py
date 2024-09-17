from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    separators=[".", "!", "?", "\n"],
)


def check_first_character(segment: str) -> bool:
    """Check if a segment needs to have the first character removed"""
    return segment[0] in (".", "!", "?")


def split_text(text: str) -> list[str]:
    """
    Split a block of text into paragraphs.

    Text is written in a single string and split using the RecursiveCharacterTextSplitter.

    This corrects for text leading with punctuation will grab the punctuation from the next segment.
    """
    text_segments = splitter.split_text(text)
    new_text_segments = []

    for i, segment in enumerate(text_segments):
        if check_first_character(segment):
            modified_new_text_segment = segment[1:].lstrip()
        else:
            modified_new_text_segment = segment

        if segment != text_segments[-1]:
            if check_first_character(text_segments[i + 1]):
                modified_new_text_segment += text_segments[i + 1][1:]

        new_text_segments.append(modified_new_text_segment)

    return new_text_segments
