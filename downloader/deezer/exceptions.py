class DeezerClientError(Exception):
    """
    Base class for exceptions that the Deezer API client raises.
    """

    pass


class DeezerAPIError(DeezerClientError):
    """
    Exception raised for errors in the Deezer API response.
    """

    pass


class DeezerTrackNotFoundError(DeezerClientError):
    """
    Exception raised when a track is not found.
    """

    pass


class DeezerURLError(DeezerClientError):
    """
    Exception raised when unable to get the track URL.
    """

    pass


class DeezerDownloadError(DeezerClientError):
    """
    Exception raised when there's an error downloading or decrypting a track.
    """

    pass
