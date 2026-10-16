
from fastapi import HTTPException, UploadFile


MAX_FILE_SIZE = 10 * 1024 * 1024
READ_CHUNK_SIZE = 64 * 1024


async def read_limited_file(file: UploadFile) -> bytes:
    content = bytearray()

    while True:
        chunk = await file.read(READ_CHUNK_SIZE)

        if not chunk:
            break

        if len(content) + len(chunk) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size must not exceed 10 MiB."
            )

        content.extend(chunk)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    return bytes(content)
