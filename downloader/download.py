import argparse
import asyncio
import json
import logging
import os

import motor.motor_asyncio
from deezer.client import DeezerClient
from deezer.exceptions import (
    DeezerDownloadError,
    DeezerTrackNotFoundError,
    DeezerURLError,
)
from motor.motor_asyncio import AsyncIOMotorCollection

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M%p",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def download_track(client: DeezerClient, isrc: str) -> bytes:
    """
    Downloads a track from Deezer using the provided ISRC.

    :param client: The DeezerClient instance.
    :param isrc: The ISRC of the track to download.
    :return: The downloaded and decrypted track as bytes.
    """
    return await client.download(isrc)


async def mass_download(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    master_key: str,
    output_folder: str,
    batch_size: int = 50,
):
    client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    while True:
        # Fetch a batch of unprocessed documents
        cursor = collection.find(
            {"$or": [{"status": {"$exists": False}}, {"status": "pending"}]},
            {"_id": 1},
        ).limit(batch_size)
        batch = await cursor.to_list(length=batch_size)

        if not batch:
            logger.info("No more tracks to download.")
            break

        logger.info(f"Processing batch of {len(batch)} tracks")
        isrcs = [doc["_id"] for doc in batch]
        deezer_client = DeezerClient(master_key)
        await process_batch(deezer_client, collection, isrcs, output_folder)


async def process_batch(
    deezer_client: DeezerClient,
    collection: AsyncIOMotorCollection,
    isrcs: list,
    output_folder: str,
):
    tasks = [
        download_and_update(deezer_client, collection, isrc, output_folder)
        for isrc in isrcs
    ]
    await asyncio.gather(*tasks)


async def download_and_update(
    deezer_client: DeezerClient,
    collection: AsyncIOMotorCollection,
    isrc: str,
    output_folder: str,
):
    try:
        track_data = await download_track(deezer_client, isrc)
        if track_data:
            isrc_folder = os.path.join(output_folder, isrc)
            os.makedirs(isrc_folder, exist_ok=True)

            # Save audio file
            audio_path = os.path.join(isrc_folder, "audio.mp3")
            with open(audio_path, "wb") as f:
                f.write(track_data)

            # Fetch lyrics from MongoDB and save as JSON
            doc = await collection.find_one({"_id": isrc})
            lyrics_path = os.path.join(isrc_folder, "lyrics.json")
            with open(lyrics_path, "w") as f:
                json.dump(
                    {
                        "unsynced": doc.get("unsynced", {}),
                        "synced": doc.get("synced", {}),
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Successfully downloaded and saved {isrc}")
            await collection.update_one(
                {"_id": isrc}, {"$set": {"status": "completed"}}
            )
        else:
            logger.warning(f"Failed to download {isrc}")
            await collection.update_one({"_id": isrc}, {"$set": {"status": "failed"}})
    except DeezerTrackNotFoundError as e:
        logger.warning(f"Track not found for ISRC {isrc}: {str(e)}")
        await collection.update_one(
            {"_id": isrc}, {"$set": {"status": "not_found", "error": str(e)}}
        )
    except DeezerURLError as e:
        logger.error(f"URL error for ISRC {isrc}: {str(e)}")
        await collection.update_one(
            {"_id": isrc}, {"$set": {"status": "url_error", "error": str(e)}}
        )
    except DeezerDownloadError as e:
        logger.error(f"Download error for ISRC {isrc}: {str(e)}")
        await collection.update_one(
            {"_id": isrc}, {"$set": {"status": "download_error", "error": str(e)}}
        )
    except Exception as e:
        logger.error(f"Unexpected error downloading {isrc}: {str(e)}")
        await collection.update_one(
            {"_id": isrc}, {"$set": {"status": "unexpected_error", "error": str(e)}}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass download tracks from Deezer")
    parser.add_argument(
        "--mongo-uri", default="mongodb://localhost:27017", help="MongoDB URI"
    )
    parser.add_argument("--db-name", required=True, help="Database name")
    parser.add_argument("--collection-name", required=True, help="Collection name")
    parser.add_argument("--master-key", required=True, help="Deezer master key")
    parser.add_argument(
        "--output-folder", required=True, help="Output folder for downloaded files"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for processing"
    )

    args = parser.parse_args()

    logger.info("Starting mass download process")
    asyncio.run(
        mass_download(
            args.mongo_uri,
            args.db_name,
            args.collection_name,
            args.master_key,
            args.output_folder,
            args.batch_size,
        )
    )
    logger.info("Mass download process completed")
