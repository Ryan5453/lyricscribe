import asyncio
import hashlib
import logging
import ssl
import time
from io import BytesIO

import httpx
from Crypto.Cipher import Blowfish
from deezer.exceptions import (
    DeezerAPIError,
    DeezerDownloadError,
    DeezerTrackNotFoundError,
    DeezerURLError,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class DeezerClient:
    """
    API client that allows for downloading music from Deezer. This interacts with both the internal and the public API.
    """

    def __init__(self, master_key: str):
        """
        Initializes the DeezerClient with the provided master key.

        :param master_key: The master key used to decrypt the track.
        """
        # Deezer's SSL keeps breaking if I don't do this...
        ssl_context = ssl.create_default_context()
        ssl_context.set_ciphers("DEFAULT@SECLEVEL=1")
        self.client = httpx.AsyncClient(timeout=15, verify=ssl_context)
        self._setup_lock = asyncio.Lock()
        self._is_setup = False
        self.api_key = None
        self.license_token = None
        self.headers = None
        self.master_key = master_key
        self.rate_limit_semaphore = asyncio.Semaphore(30)
        self.last_request_time = time.time()

    async def _internal_request(self, params: dict, data: dict) -> dict:
        """
        Makes an internal request to the internal Deezer API.

        :param params: The parameters to send with the request.
        :param data: The data to send with the request.
        :return: The response from the Deezer API.
        """
        await self._ensure_setup()
        response = await self.client.post(
            "https://www.deezer.com/ajax/gw-light.php",
            params=params,
            json=data,
            headers={"Accept": "application/json"},
        )
        if response.status_code != 200:
            raise DeezerAPIError(
                f"Error requesting Deezer gw endpoint. Status code: {response.status_code}"
            )
        return response.json()

    async def _ensure_setup(self):
        """
        Ensures that the API client is set up correctly.
        """
        async with self._setup_lock:
            if not self._is_setup:
                await self._setup()

    async def _setup(self):
        """
        Sets up the API client by creating a session and getting the necessary tokens.
        """
        logger.info("Setting up DeezerClient")
        json = (
            await self.client.post(
                "https://www.deezer.com/ajax/gw-light.php",
                params={
                    "method": "deezer.getUserData",
                    "input": 3,
                    "api_version": 1.0,
                    "api_token": "",
                },
                json={},
                headers={"Accept": "application/json"},
            )
        ).json()
        self.api_key = json["results"]["checkForm"]
        self.license_token = json["results"]["USER"]["OPTIONS"]["license_token"]
        self.headers = {"cookie": f"sid={json['results']['SESSION_ID']}"}
        self._is_setup = True
        logger.info("DeezerClient setup completed")

    async def _get_id_from_isrc(self, isrc: str) -> int:
        """
        Gets the ID of a track from its ISRC.

        :param isrc: The ISRC of the track.
        :return: The ID of the track.
        """
        await self._ensure_setup()
        response = await self.client.get(
            f"https://api.deezer.com/2.0/track/isrc:{isrc}"
        )
        if response.status_code != 200:
            raise DeezerAPIError(
                f"Error requesting Deezer API. Status code: {response.status_code}"
            )
        json = response.json()
        if "error" in json:
            raise DeezerTrackNotFoundError(
                f"Track with ISRC {isrc} not found on Deezer."
            )
        return json["id"]

    async def _get_track(self, id: int):
        """
        Gets the track data from the Deezer API. This contains the track token.

        :param id: The ID of the track.
        :return: The track data.
        """
        await self._ensure_setup()
        data = {
            "sng_id": id,
        }
        params = {
            "method": "song.getData",
            "input": 3,
            "api_version": 1.0,
            "api_token": self.api_key,
        }
        response = await self.client.post(
            "https://www.deezer.com/ajax/gw-light.php",
            params=params,
            json=data,
            headers=self.headers,
        )
        if response.status_code != 200:
            raise DeezerAPIError(
                f"Error getting track data. Status code: {response.status_code}"
            )
        json = response.json()
        if "results" not in json:
            raise DeezerTrackNotFoundError(f"Track with ID {id} not found on Deezer.")
        return json

    async def _get_url(self, track_token: str) -> str:
        """
        Gets the URL of a track from its track token.

        :param track_token: The track token of the track.
        :return: The URL of the track.
        """
        await self._ensure_setup()
        data = {
            "license_token": self.license_token,
            "media": [
                {
                    "type": "FULL",
                    "formats": [{"cipher": "BF_CBC_STRIPE", "format": "MP3_128"}],
                }
            ],
            "track_tokens": [track_token],
        }
        response = await self.client.post(
            "https://media.deezer.com/v1/get_url", json=data
        )
        if response.status_code != 200:
            raise DeezerAPIError(
                f"Error getting track URL. Status code: {response.status_code}"
            )
        json = response.json()
        if "errors" in json["data"][0]:
            raise DeezerURLError(
                f"Could not get Deezer URL. Error: {json['data'][0]['errors']}"
            )
        types = json["data"][0]["media"]
        if not types:
            raise DeezerURLError(f"Could not find download URL in response.")
        return types[0]["sources"][0]["url"]

    def _generate_blowfish_key(self, track_id: int) -> bytes:
        """
        Generates the blowfish key for a track.

        :param track_id: The ID of the track.
        :return: The blowfish key.
        """
        m = hashlib.md5()
        m.update(bytes([ord(x) for x in str(track_id)]))
        id_md5 = m.hexdigest()

        blowfish_key = bytes(
            (
                [
                    (ord(id_md5[i]) ^ ord(id_md5[i + 16]) ^ ord(self.master_key[i]))
                    for i in range(16)
                ]
            )
        )

        return blowfish_key

    def _decrypt_chunk(self, data: bytes, blowfish_key: bytes) -> bytes:
        """
        Decrypts a chunk of data using the provided blowfish key.

        :param data: The data to decrypt.
        :param blowfish_key: The blowfish key to use for decryption.
        :return: The decrypted data.
        """
        cipher = Blowfish.new(
            blowfish_key, Blowfish.MODE_CBC, bytes([i for i in range(8)])
        )
        return cipher.decrypt(data)

    async def _decrypt_track(self, url: str, track_id: int) -> bytes:
        """
        Decrypts a track from the provided URL.

        :param url: The URL of the track.
        :param track_id: The ID of the track.
        :return: The decrypted track.
        """
        await self._ensure_setup()
        blowfish_key = self._generate_blowfish_key(track_id)
        decrypted_data = BytesIO()
        async with self.client.stream("GET", url) as response:
            if response.status_code != 200:
                raise DeezerDownloadError(
                    f"Error downloading track. Status code: {response.status_code}"
                )
            iterations = 0
            async for data in response.aiter_bytes(chunk_size=2048):
                if iterations % 3 == 0 and len(data) == 2048:
                    data = self._decrypt_chunk(data, blowfish_key)
                decrypted_data.write(data)
                iterations += 1
        return decrypted_data.getvalue()

    async def download(self, isrc: str) -> bytes:
        """
        Downloads and decrypts a track from Deezer using its ISRC.

        :param isrc: The ISRC of the track.
        :return: The decrypted track.
        """
        async with self.rate_limit_semaphore:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < 0.1:
                await asyncio.sleep(0.1 - time_since_last_request)

            self.last_request_time = time.time()

            logger.debug(f"Starting download for ISRC: {isrc}")
            track_id = await self._get_id_from_isrc(isrc)
            track = await self._get_track(track_id)
            url = await self._get_url(track["results"]["TRACK_TOKEN"])
            result = await self._decrypt_track(url, track_id)
            logger.debug(
                f"Successfully downloaded and decrypted track for ISRC: {isrc}"
            )
            return result
