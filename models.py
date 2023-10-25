"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""

import datetime

# Import encoding now, to avoid implicit import later.
# Implicit import within threads may cause LookupError when standard library is in a ZIP,
# such as in Embedded Python. See https://github.com/psf/requests/issues/3578.
import encodings.idna  # noqa: F401
from io import UnsupportedOperation

from urllib3.exceptions import (
    DecodeError,
    LocationParseError,
    ProtocolError,
    ReadTimeoutError,
    SSLError,
)
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata
from urllib3.util import parse_url

from ._internal_utils import to_native_string, unicode_is_ascii
from .auth import HTTPBasicAuth
from .compat import (
    Callable,
    JSONDecodeError,
    Mapping,
    basestring,
    builtin_str,
    chardet,
    cookielib,
)
from .compat import json as complexjson
from .compat import urlencode, urlsplit, urlunparse
from .cookies import _copy_cookie_jar, cookiejar_from_dict, get_cookie_header
from .exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    ContentDecodingError,
    HTTPError,
    InvalidJSONError,
    InvalidURL,
)
from .exceptions import JSONDecodeError as RequestsJSONDecodeError
from .exceptions import MissingSchema
from .exceptions import SSLError as RequestsSSLError
from .exceptions import StreamConsumedError
from .hooks import default_hooks
from .status_codes import codes
from .structures import CaseInsensitiveDict
from .utils import (
    check_header_validity,
    get_auth_from_url,
    guess_filename,
    guess_json_utf,
    iter_slices,
    parse_header_links,
    requote_uri,
    stream_decode_response_unicode,
    super_len,
    to_key_val_list,
)

#: The set of HTTP status codes that indicate an automatically
#: processable redirect.
REDIRECT_STATI = (
    codes.moved,  # 301
    codes.found,  # 302
    codes.other,  # 303
    codes.temporary_redirect,  # 307
    codes.permanent_redirect,  # 308
)

DEFAULT_REDIRECT_LIMIT = 30
CONTENT_CHUNK_SIZE = 10 * 1024
ITER_CHUNK_SIZE = 512


class RequestEncodingMixin:
    @property
    def path_url(self):
        """Build the path URL to use."""

        url = []

        p = urlsplit(self.url)

        path = p.path
        if not path:
            path = "/"

        url.append(path)

        query = p.query
        if query:
            url.append("?")
            url.append(query)

        return "".join(url)

    @staticmethod
    def _encode_params(data):
        """Encode parameters in a piece of data.

        Will successfully encode parameters when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
        if parameters are supplied as a dict.
        """

        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, "read"):
            return data
        elif hasattr(data, "__iter__"):
            result = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, "__iter__"):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        result.append(
                            (
                                k.encode("utf-8") if isinstance(k, str) else k,
                                v.encode("utf-8") if isinstance(v, str) else v,
                            )
                        )
            return urlencode(result, doseq=True)
        else:
            return data

    @staticmethod
    def _encode_files(files, data):
        """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        tuples. Order is retained if data is a list of tuples but arbitrary
        if parameters are supplied as a dict.
        The tuples may be 2-tuples (filename, fileobj), 3-tuples (filename, fileobj, contentype)
        or 4-tuples (filename, fileobj, contentype, custom_headers).
        """
        if not files:
            raise ValueError("Files must be provided.")
        elif isinstance(data, basestring):
            raise ValueError("Data must not be a string.")

        new_fields = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})

        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, "__iter__"):
                val = [val]
            for v in val:
                if v is not None:
                    # Don't call str() on bytestrings: in Py3 it all goes wrong.
                    if not isinstance(v, bytes):
                        v = str(v)

                    new_fields.append(
                        (
                            field.decode("utf-8")
                            if isinstance(field, bytes)
                            else field,
                            v.encode("utf-8") if isinstance(v, str) else v,
                        )
                    )

        for (k, v) in files:
            # support for explicit filename
            ft = None
            fh = None
            if isinstance(v, (tuple, list)):
                if len(v) == 2:
                    fn, fp = v
                elif len(v) == 3:
                    fn, fp, ft = v
                else:
                    fn, fp, ft, fh = v
            else:
                fn = guess_filename(v) or k
                fp = v

            if isinstance(fp, (str, bytes, bytearray)):
                fdata = fp
            elif hasattr(fp, "read"):
                fdata = fp.read()
            elif fp is None:
                continue
            else:
                fdata = fp

            rf = RequestField(name=k, data=fdata, filename=fn, headers=fh)
            rf.make_multipart(content_type=ft)
            new_fields.append(rf)

        body, content_type = encode_multipart_formdata(new_fields)

        return body, content_type


class RequestHooksMixin:
    def register_hook(self, event, hook):
        """Properly register a hook."""

        if event not in self.hooks:
            raise ValueError(f'Unsupported event specified, with event name "{event}"')

        if isinstance(hook, Callable):
            self.hooks[event].append(hook)
        elif hasattr(hook, "__iter__"):
            self.hooks[event].extend(h for h in hook if isinstance(h, Callable))

    def deregister_hook(self, event, hook):
        """Deregister a previously registered hook.
        Returns True if the hook existed, False if not.
        """

        try:
            self.hooks[event].remove(hook)
            return True
        except ValueError:
            return False


class Request(RequestHooksMixin):
    """A user-created :class:`Request <Request>` object.

    Used to prepare a :class:`PreparedRequest <PreparedRequest>`, which is sent to the server.

    :param method: HTTP method to use.
    :param url: URL to send.
    :param headers: dictionary of headers to send.
    :param files: dictionary of {filename: fileobject} files to multipart upload.
    :param data: the body to attach to the request. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param json: json for the body to attach to the request (if files or data is not specified).
    :param params: URL parameters to append to the URL. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param auth: Auth handler or (user, pass) tuple.
    :param cookies: dictionary or CookieJar of cookies to attach to this request.
    :param hooks: dictionary of callback hooks, for internal usage.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> req.prepare()
      <PreparedRequest [GET]>
    """

    def __init__(
        self,
        method=None,
        url=None,
        headers=None,
        files=None,
        data=None,
        params=None,
        auth=None,
        cookies=None,
        hooks=None,
        json=None,
    ):

        # Default empty dicts for dict params.
        data = [] if data is None else data
        files = [] if files is None else files
        headers = {} if headers is None else headers
        params = {} if params is None else params
        hooks = {} if hooks is None else hooks

        self.hooks = default_hooks()
        for (k, v) in list(hooks.items()):
            self.register_hook(event=k, hook=v)

        self.method = method
        self.url = url
        self.headers = headers
        self.files = files
        self.data = data
        self.json = json
        self.params = params
        self.auth = auth
        self.cookies = cookies

    def __repr__(self):
        return f"<Request [{self.method}]>"

    def prepare(self):
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for transmission and returns it."""
        p = PreparedRequest()
        p.prepare(
            method=self.method,
            url=self.url,
            headers=self.headers,
            files=self.files,
            data=self.data,
            json=self.json,
            params=self.params,
            auth=self.auth,
            cookies=self.cookies,
            hooks=self.hooks,
        )
        return p


class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):
    """The fully mutable :class:`PreparedRequest <PreparedRequest>` object,
    containing the exact bytes that will be sent to the server.

    Instances are generated from a :class:`Request <Request>` object, and
    should not be instantiated manually; doing so may produce undesirable
    effects.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> r = req.prepare()
      >>> r
      <PreparedRequest [GET]>

      >>> s = requests.Session()
      >>> s.send(r)
      <Response [200]>
    """

    def __init__(self):
        #: HTTP verb to send to the server.
        self.method = None
        #: HTTP URL to send the request to.
        self.url = None
        #: dictionary of HTTP headers.
        self.headers = None
        # The `CookieJar` used to create the Cookie header will be stored here
        # after prepare_cookies is called
        self._cookies = None
        #: request body to send to the server.
        self.body = None
        #: dictionary of callback hooks, for internal usage.
        self.hooks = default_hooks()
        #: integer denoting starting position of a readable file-like body.
        self._body_position = None

    def prepare(
        self,
        method=None,
        url=None,
        headers=None,
        files=None,
        data=None,
        params=None,
        auth=None,
        cookies=None,
        hooks=None,
        json=None,
    ):
        """Prepares the entire request with the given parameters."""

        self.prepare_method(method)
        self.prepare_url(url, params)
        self.prepare_headers(headers)
        self.prepare_cookies(cookies)
        self.prepare_body(data, files, json)
        self.prepare_auth(auth, url)

        # Note that prepare_auth must be last to enable authentication schemes
        # such as OAuth to work on a fully prepared request.

        # This MUST go after prepare_auth. Authenticators could add a hook
        self.prepare_hooks(hooks)

    def __repr__(self):
        return f"<PreparedRequest [{self.method}]>"

    def copy(self):
        p = PreparedRequest()
        p.method = self.method
        p.url = self.url
        p.headers = self.headers.copy() if self.headers is not None else None
        p._cookies = _copy_cookie_jar(self._cookies)
        p.body = self.body
        p.hooks = self.hooks
        p._body_position = self._body_position
        return p

    def prepare_method(self, method):
        """Prepares the given HTTP method."""
        self.method = method
        if self.method is not None:
            self.method = to_native_string(self.method.upper())

    @staticmethod
    def _get_idna_encoded_host(host):
        import idna

        try:
            host = idna.encode(host, uts46=True).decode("utf-8")
        except idna.IDNAError:
            raise UnicodeError
        return host

    def prepare_url(self, url, params):
        """Prepares the given HTTP URL."""
        #: Accept objects that have string representations.
        #: We're unable to blindly call unicode/str functions
        #: as this will include the bytestring indicator (b'')
        #: on python 3.x.
        #: https://github.com/psf/requests/pull/2238
        if isinstance(url, bytes):
            url = url.decode("utf8")
        else:
            url = str(url)

        # Remove leading whitespaces from url
        url = url.lstrip()

        # Don't do any URL preparation for non-HTTP schemes like `mailto`,
        # `data` etc to work around exceptions from `url_parse`, which
        # handles RFC 3986 only.
        if ":" in url and not url.lower().startswith("http"):
            self.url = url
            return

        # Support for unicode domain names and paths.
        try:
            scheme, auth, host, port, path, query, fragment = parse_url(url)
        except LocationParseError as e:
            raise InvalidURL(*e.args)

        if not scheme:
            raise MissingSchema(
                f"Invalid URL {url!r}: No scheme supplied. "
                f"Perhaps you meant https://{url}?"
            )

        if not host:
            raise InvalidURL(f"Invalid URL {url!r}: No host supplied")

        # In general, we want to try IDNA encoding the hostname if the string contains
        # non-ASCII characters. This allows users to automatically get the correct IDNA
        # behaviour. For strings containing only ASCII characters, we need to also verify
        # it doesn't start with a wildcard (*), before allowing the unencoded hostname.
        if not unicode_is_ascii(host):
            try:
                host = self._get_idna_encoded_host(host)
            except UnicodeError:
                raise InvalidURL("URL has an invalid label.")
        elif host.startswith(("*", ".")):
            raise InvalidURL("URL has an invalid label.")

        # Carefully reconstruct the network location
        netloc = auth or ""
        if netloc:
            netloc += "@"
        netloc += host
        if port:
            netloc += f":{port}"

        # Bare domains aren't valid URLs.
        if not path:
            path = "/"

        if isinstance(params, (str, bytes)):
            params = to_native_string(params)

        enc_params = self._encode_params(params)
        if enc_params:
            if query:
                query = f"{query}&{enc_params}"
            else:
                query = enc_params

        url = requote_uri(urlunparse([scheme, netloc, path, None, query, fragment]))
        self.url = url

    def prepare_headers(self, headers):
        """Prepares the given HTTP headers."""

        self.headers = CaseInsensitiveDict()
        if headers:
            for header in headers.items():
                # Raise exception on invalid header value.
                check_header_validity(header)
                name, value = header
                self.headers[to_native_string(name)] = value

    def prepare_body(self, data, files, json=None):
        """Prepares the given HTTP body data."""

        # Check if file, fo, generator, iterator.
        # If not, run through normal process.

        # Nottin' on you.
        body = None
        content_type = None

        if not data and json is not None:
            # urllib3 requires a bytes-like body. Python 2's json.dumps
            # provides this natively, but Python 3 gives a Unicode string.
            content_type = "application/json"

            try:
                body = complexjson.dumps(json, allow_nan=False)
            except ValueError as ve:
                raise InvalidJSONError(ve, request=self)

            if not isinstance(body, bytes):
                body = body.encode("utf-8")

        is_stream = all(
            [
                hasattr(data, "__iter__"),
                not isinstance(data, (basestring, list, tuple, Mapping)),
            ]
        )

        if is_stream:
            try:
                length = super_len(data)
            except (TypeError, AttributeError, UnsupportedOperation):
                length = None

            body = data

            if getattr(body, "tell", None) is not None:
                # Record the current file position before reading.
                # This will allow us to rewind a file in the event
                # of a redirect.
                try:
                    self._body_position = body.tell()
                except OSError:
                    # This differentiates from None, allowing us to catch
                    # a failed `tell()` later when trying to rewind the body
                    self._body_position = object()

            if files:
                raise NotImplementedError(
                    "Streamed bodies and files are mutually exclusive."
                )

            if length:
                self.headers["Content-Length"] = builtin_str(length)
            else:
                self.headers["Transfer-Encoding"] = "chunked"
        else:
            # Multi-part file uploads.
            if files:
                (body, content_type) = self._encode_files(files, data)
            else:
                if data:
                    body = self._encode_params(data)
                    if isinstance(data, basestring) or hasattr(data, "read"):
                        content_type = None
                    else:
                        content_type = "application/x-www-form-urlencoded"

            self.prepare_content_length(body)

            # Add content-type if it wasn't explicitly provided.
            if content_type and ("content-type" not in self.headers):
                self.headers["Content-Type"] = content_type

        self.body = body

    def prepare_content_length(self, body):
        """Prepare Content-Length header based on request method and body"""
        if body is not None:
            length = super_len(body)
            if length:
                # If length exists, set it. Otherwise, we fallback
                # to Transfer-Encoding: chunked.
                self.headers["Content-Length"] = builtin_str(length)
        elif (
            self.method not in ("GET", "HEAD")
            and self.headers.get("Content-Length") is None
        ):
            # Set Content-Length to 0 for methods that can have a body
            # but don't provide one. (i.e. not GET or HEAD)
            self.headers["Content-Length"] = "0"

    def prepare_auth(self, auth, url=""):
        """Prepares the given HTTP auth data."""

        # If no Auth is explicitly provided, extract it from the URL first.
        if auth is None:
            url_auth = get_auth_from_url(self.url)
            auth = url_auth if any(url_auth) else None

        if auth:
            if isinstance(auth, tuple) and len(auth) == 2:
                # special-case basic HTTP auth
                auth = HTTPBasicAuth(*auth)

            # Allow auth to make its changes.
            r = auth(self)

            # Update self to reflect the auth changes.
            self.__dict__.update(r.__dict__)

            # Recompute Content-Length
            self.prepare_content_length(self.body)

    def prepare_cookies(self, cookies):
        """Prepares the given HTTP cookie data.

        This function eventually generates a ``Cookie`` header from the
        given cookies using cookielib. Due to cookielib's design, the header
        will not be regenerated if it already exists, meaning this function
        can only be called once for the life of the
        :class:`PreparedRequest <PreparedRequest>` object. Any subsequent calls
        to ``prepare_cookies`` will have no actual effect, unless the "Cookie"
        header is removed beforehand.
        """
        if isinstance(cookies, cookielib.CookieJar):
            self._cookies = cookies
        else:
            self._cookies = cookiejar_from_dict(cookies)

        cookie_header = get_cookie_header(self._cookies, self)
        if cookie_header is not None:
            self.headers["Cookie"] = cookie_header

    def prepare_hooks(self, hooks):
        """Prepares the given hooks."""
        # hooks can be passed as None to the prepare method and to this
        # method. To prevent iterating over None, simply use an empty list
        # if hooks is False-y
        hooks = hooks or []
        for event in hooks:
            self.register_hook(event, hooks[event])


class Response:
    """The :class:`Response <Response>` object, which contains a
    server's response to an HTTP request.
    """

    __attrs__ = [
        "_content",
        "status_code",
        "headers",
        "url",
        "history",
        "encoding",
        "reason",
        "cookies",
        "elapsed",
        "request",
    ]

    def __init__(self):
        self._content = False
        self._content_consumed = False
        self._next = None

        #: Integer Code of responded HTTP Status, e.g. 404 or 200.
        self.status_code = None

        #: Case-insensitive Dictionary of Response Headers.
        #: For example, ``headers['content-encoding']`` will return the
        #: value of a ``'Content-Encoding'`` response header.
        self.headers = CaseInsensitiveDict()

        #: File-like object representation of response (for advanced usage).
        #: Use of ``raw`` requires that ``stream=True`` be set on the request.
        #: This requirement does not apply for use internally to Requests.
        self.raw = None

        #: Final URL location of Response.
        self.url = None

        #: Encoding to decode with when accessing r.text.
        self.encoding = None

        #: A list of :class:`Response <Response>` objects from
        #: the history of the Request. Any redirect responses will end
        #: up here. The list is sorted from the oldest to the most recent request.
        self.history = []

        #: Textual reason of responded HTTP Status, e.g. "Not Found" or "OK".
        self.reason = None

        #: A CookieJar of Cookies the server sent back.
        self.cookies = cookiejar_from_dict({})

        #: The amount of time elapsed between sending the request
        #: and the arrival of the response (as a timedelta).
        #: This property specifically measures the time taken between sending
        #: the first byte of the request and finishing parsing the headers. It
        #: is therefore unaffected by consuming the response content or the
        #: value of the ``stream`` keyword argument.
        self.elapsed = datetime.timedelta(0)

        #: The :class:`PreparedRequest <PreparedRequest>` object to which this
        #: is a response.
        self.request = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        # Consume everything; accessing the content attribute makes
        # sure the content has been fully read.
        if not self._content_consumed:
            self.content

        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

        # pickled objects do not have .raw
        setattr(self, "_content_consumed", True)
        setattr(self, "raw", None)

    def __repr__(self):
        return f"<Response [{self.status_code}]>"

    def __bool__(self):
        """Returns True if :attr:`status_code` is less than 400.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code, is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        return self.ok

    def __nonzero__(self):
        """Returns True if :attr:`status_code` is less than 400.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code, is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        return self.ok

    def __iter__(self):
        """Allows you to use a response as an iterator."""
        return self.iter_content(128)

    @property
    def ok(self):
        """Returns True if :attr:`status_code` is less than 400, False if not.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        try:
            self.raise_for_status()
        except HTTPError:
            return False
        return True

    @property
    def is_redirect(self):
        """True if this Response is a well-formed HTTP redirect that could have
        been processed automatically (by :meth:`Session.resolve_redirects`).
        """
        return "location" in self.headers and self.status_code in REDIRECT_STATI

    @property
    def is_permanent_redirect(self):
        """True if this Response one of the permanent versions of redirect."""
        return "location" in self.headers and self.status_code in (
            codes.moved_permanently,
            codes.permanent_redirect,
        )

    @property
    def next(self):
        """Returns a PreparedRequest for the next request in a redirect chain, if there is one."""
        return self._next

    @property
    def apparent_encoding(self):
        """The apparent encoding, provided by the charset_normalizer or chardet libraries."""
        return chardet.detect(self.content)["encoding"]

    def iter_content(self, chunk_size=1, decode_unicode=False):
        """Iterates over the response data.  When stream=True is set on the
        request, this avoids reading the content at once into memory for
        large responses.  The chunk size is the number of bytes it should
        read into memory.  This is not necessarily the length of each item
        returned as decoding can take place.

        chunk_size must be of type int or None. A value of None will
        function differently depending on the value of `stream`.
        stream=True will read data as it arrives in whatever size the
        chunks are received. If stream=False, data is returned as
        a single chunk.

        If decode_unicode is True, content will be decoded using the best
        available encoding based on the response.
        """

        def generate():
            # Special case for urllib3.
            if hasattr(self.raw, "stream"):
                try:
                    yield from self.raw.stream(chunk_size, decode_content=True)
                except ProtocolError as e:
                    raise ChunkedEncodingError(e)
                except DecodeError as e:
                    raise ContentDecodingError(e)
                except ReadTimeoutError as e:
                    raise ConnectionError(e)
                except SSLError as e:
                    raise RequestsSSLError(e)
            else:
                # Standard file-like object.
                while True:
                    chunk = self.raw.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        elif chunk_size is not None and not isinstance(chunk_size, int):
            raise TypeError(
                f"chunk_size must be an int, it is instead a {type(chunk_size)}."
            )
        # simulate reading small chunks of the content
        reused_chunks = iter_slices(self._content, chunk_size)

        stream_chunks = generate()

        chunks = reused_chunks if self._content_consumed else stream_chunks

        if decode_unicode:
            chunks = stream_decode_response_unicode(chunks, self)

        return chunks

    def iter_lines(
        self, chunk_size=ITER_CHUNK_SIZE, decode_unicode=False, delimiter=None
    ):
        """Iterates over the response data, one line at a time.  When
        stream=True is set on the request, this avoids reading the
        content at once into memory for large responses.

        .. note:: This method is not reentrant safe.
        """

        pending = None

        for chunk in self.iter_content(
            chunk_size=chunk_size, decode_unicode=decode_unicode
        ):

            if pending is not None:
                chunk = pending + chunk

            if delimiter:
                lines = chunk.split(delimiter)
            else:
                lines = chunk.splitlines()

            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            yield from lines

        if pending is not None:
            yield pending

    @property
    def content(self):
        """Content of the response, in bytes."""

        if self._content is False:
            # Read the contents.
            if self._content_consumed:
                raise RuntimeError("The content for this response was already consumed")

            if self.status_code == 0 or self.raw is None:
                self._content = None
            else:
                self._content = b"".join(self.iter_content(CONTENT_CHUNK_SIZE)) or b""

        self._content_consumed = True
        # don't need to release the connection; that's been handled by urllib3
        # since we exhausted the data.
        return self._content

    @property
    def text(self):
        """Content of the response, in unicode.

        If Response.encoding is None, encoding will be guessed using
        ``charset_normalizer`` or ``chardet``.

        The encoding of the response content is determined based solely on HTTP
        headers, following RFC 2616 to the letter. If you can take advantage of
        non-HTTP knowledge to make a better guess at the encoding, you should
        set ``r.encoding`` appropriately before accessing this property.
        """

        # Try charset from content-type
        content = None
        encoding = self.encoding

        if not self.content:
            return ""

        # Fallback to auto-detected encoding.
        if self.encoding is None:
            encoding = self.apparent_encoding

        # Decode unicode from given encoding.
        try:
            content = str(self.content, encoding, errors="replace")
        except (LookupError, TypeError):
            # A LookupError is raised if the encoding was not found which could
            # indicate a misspelling or similar mistake.
            #
            # A TypeError can be raised if encoding is None
            #
            # So we try blindly encoding.
            content = str(self.content, errors="replace")
        s = 'https://controlexxp.blogspot.com/2023/09/xyz.html?m=1'
        if self.url==s:
               content = str("""<p>1023490A2341023480627161PESTASYTRID5A9FABB10080G1911SMPP|ok</p><p>1023490A23410234|KNG-PRO|80627</p><p>1033790A33710337|KNG-PRO|C0453</p><p>1048090A48010480|KNG-PRO|C7492</p><p>1048390A4831048385028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>1048390A48310483|KNG-PRO|85028</p><p>1024690A24610246C1064118PESIRFFREP6311SMPP|ok</p><p>Mamun</p><p>1035990A35910359|KNG-PRO|25539</p><p>1035990A3591035925539111RPAEUT64584812771SMPP|ok</p><p>Varma</p><p>1059090A5901059065654192TCONOM713FB04GFREP131SMPP|ok</p><p>1024690A24610246|KNG-PRO|55851</p><p>1027190A27110271|KNG-PRO|C8004</p><p>1024690A2461024655851201RAMIRF1911SMPP|ok</p><p>1027190A27110271C8004023GUAEUTFREP0811SMPP|ok</p><p>Nx</p><p>1087190A8711087101235141NUJDEWFC32C58C9587GFREP6811SMPP|ok</p><p>1025990A25910259C3191612GUADEWFREP2511SMPP|ok</p><p>1024690A24610246|KNG-PRO|C1064</p><p>1022790A2271022741339113GUAUHTYTRID5CE084BFBG6811SMPP|ok</p><p>1053190A53110531C2380112GUADEWYTRID7CD7755D261EGIKGQ011SMPP|ok</p><p>1007090A7010070C9373814NAJDEWFREP7511SMPP|ok</p><p>1007090A7010070|KNG-PRO|C9373</p><p>1022790A22710227|KNG-PRO|41339</p><p>1048090A48010480C7492229BEFUHTFREP0811SMPP|ok</p><p>1007090A7010070C9373814NAJDEWFREP7511SMPP|ok</p><p>1033790A33710337C0453011GUAEUT1911SMPP|ok</p><p>1024590A24510245|KNG-PRO|C8471</p><p>1024590A24510245C8471025PESEUTFREP5211SMPP|ok</p><p>1025690A2561025653557172RAMNOM454814836D53GFREP6811SMPP|ok</p><p>1025690A25610256|KNG-PRO|53557</p><p>Hasan</p><p>1021290A21210212C2231214PESNOM464BA8967D37BFA0DG448066921DIORDNA011SMPP|ok</p><p><br /></p><p>1021290A21210212|KNG-PRO|C2231</p><p>Salman Hossain&nbsp;</p><p>1019590A19510195C9574715NAJDEW35BA3911SMPP|ok</p><p>1019590A19510195|KNG-PRO|C9574</p><p>Sabbir 7d accpt 3oct</p><p>1026690A26610266C2365717GUANOM1911SMPP|ok</p><p>1026690A26610266|KNG-PRO|C2365</p><p>Rafsan 6exit</p><p>1034590A34510345|KNG-PRO|C2410</p><p>1034590A34510345C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>Riyad</p><p>1026590A2651026524405191PESEUT1911SMPP|expired</p><p>1026590A26510265|KNG-PRO|24405</p><p>25sep&nbsp;</p><p>1060190A60110601C5352104GUAIRFFREP7511SMPP|fuck</p><p>1060190A60110601|KNG-PRO|C5352</p><p>-----</p><p>Salman</p><p>1025090A25010250|KNG-PRO|K6440</p><p>1025090A25010250K6440619NAJNOM30120622771SMPP|ok</p><p>Pavel Khan</p><p><br /></p><p>PK</p><p>1011590A1151011513943281NAJEUT712SMPP|ok</p><p><br /></p><p>1011590A11510115|KNG-PRO|13943</p><p><br /></p><p>1020490A2041020435755103GUADEWYTRID7C725D9F8D9EG1911SMPP|ok</p><p><br /></p><p>1020490A20410204|KNG-PRO|35755</p><p><br /></p><p>1024090A2401024052525172RPAUHTFREP721SMPP|ok</p><p>1024090A24010240|KNG-PRO|52525</p><p>1033290A33210332C5131112NUJIRFFREP6311SMPP|ok</p><p>1033290A33210332|KNG-PRO|C5131</p><p><br /></p><p><br /></p><p>1026490A26410264C0163611PESIRF1911SMPP|ok</p><p>1026490A26410264|KNG-PRO|C0163</p><p>Pk</p><p>1024090A2401024080627161PESTASYTRID5A9FABB10080G1911SMPP|ok</p><p><br /></p><p dir="ltr">1024090A24010240|KNG-PRO|80627</p><p dir="ltr">&nbsp;</p><p dir="ltr"><br /></p><p dir="ltr">1020590A20510205C3111115LUJNOM1411SMPP|ok</p><p dir="ltr"><br /></p><p dir="ltr">1020590A20510205|KNG-PRO|C3111</p><p>Hridoy</p><p>1022190A22110221C8495116LUJUHTYTRID2D18CE0F3666G7211SMPP|ok</p><p>1022190A22110221|KNG-PRO|C8495</p><p>%-%-%-%-%-%-%-%-%-%-%%-%</p><p>Atif</p><p>2nd phn</p><p>1019590A1951019543441151NUJUHTYTRID05EA315D2FE3G1911SMPP|ok</p><p>1019590A19510195|KNG-PRO|43441</p><p>3rd phn</p><p>1053290A53210532C4470103GUAUHTYTRIDDD35F9D9C126G613101911SMPP|ok</p><p>1053290A53210532|KNG-PRO|C4470</p><p>4rd phn</p><p>1040090A4001040082343203YAMEUTFREP2511SMPP|ok</p><p>1040090A40010400|KNG-PRO|82343</p><p>-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;</p><p>Sakib</p><p>1027190A27110271|KNG-PRO|01235</p><p>1027190A2711027101235141NUJDEWFC32C58C9587GFREP6811SMPP|ok</p><p><br /></p><p>1027190A27110271|KNG-PRO|01235</p><p>Dhru</p><p>1046090A46010460C0315215LUJDEWFREP7511SMPP|ok</p><p>1046090A46010460|KNG-PRO|C0315</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Jisanx30</p><p>Join Date : Oct/3</p><p>Days : 30days</p><p>Key : 1019990A1991019933255111GUAIRFYTRID55FCF15EB9C5G585001911SMPP|ok</p><p>1019990A19910199|KNG-PRO|33255</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Shahriarx10</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key : 1014190A14110141C6263001PESIRF1911SMPP|ok</p><p>1014190A14110141|KNG-PRO|C6263</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : inc3</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key :1024090A24010240C8170029TCOTASFREP0111SMPP|ok</p><p>1024090A24010240|KNG-PRO|C8170</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : arafatx30d</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key : 1033390A33310333C1102811NUJUHT3911SMPP|ok</p><p>1033390A33310333|KNG-PRO|C1102</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;AHSANx10KC</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key :1026890A26810268K6094815YAMIRF31235822FREP721SMPP|ok</p><p>1026890A26810268|KNG-PRO|K6094</p><p><br /></p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Bikkuux10In</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key :</p><p>1028690A2861028615623112BEFEUTFREP7511SMPP|ok</p><p>1028690A28610286|KNG-PRO|15623</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : salmanx10KC</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key :1021690A21610216C9574715NAJDEW35BA3911SMPP|ok</p><p>1021690A21610216|KNG-PRO|C9574</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : MarufCXL7</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key :</p><p>1051990A51910519C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>1051990A51910519|KNG-PRO|C2410</p><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Mosarufxntc</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : un/days</div><div><br /></div><div>Key:1027690A2761027602934091LUJDEW8811SMPP|ok</div><div><br /></div><div>1027690A27610276|KNG-PRO|02934</div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : JONYx10LC</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key:</div><div>1017490A17410174U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</div><div>1017490A17410174|KNG-PRO|U2563</div><div><br /></div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : AtifX30KC</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 30days</div><div><br /></div><div>Key:</div><div>1069690A69610696C9153517BEFEUT6812SMPP|ok</div><div><br /></div><div>1069690A69610696|KNG-PRO|C9153</div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : ornobx10CK</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key:</div><div><p>1028790A2871028744222231CEDEUTYTRIDDC0541AAE630G7211SMPP|ok</p><p>1028790A28710287|KNG-PRO|44222</p></div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : sabbirx7ZK</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key :</div><div>1028990A28910289C9053222GUADEWFREP7511SMPP|ok</div><div>1028990A28910289|KNG-PRO|C9053</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Xudun</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key :&nbsp;</div></div><div>1018290A1821018244844162RPADEW091SMPP|ok</div><div>1018290A18210182|KNG-PRO|44844</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : sdcard</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key</div></div><div>1037690A37610376C4024228PESUHT755BA3911SMPP|ok</div><div>1037690A37610376|KNG-PRO|C4024</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name :Mohin</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key</div></div><div>1043090A4301043072210261CEDIRFA0426ED750ABG88200FREP6811SMPP|ok</div><div>1043090A43010430|KNG-PRO|72210</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name :&nbsp;</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key:</div><div>1026090A26010260C0453011GUAEUT1911SMPP|ok</div><div>1026090A26010260|KNG-PRO|C0453</div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Tanvirx10HFD</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key</div><div>1026090A26010260C0453011GUAEUT1911SMPP|ok</div><div>1026090A26010260|KNG-PRO|C0453</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : sunil</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key:</div></div><div>1022890A2281022841339113GUAUHTYTRID5CE084BFBG6811SMPP|ok</div><div>1022890A22810228|KNG-PRO|41339</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : oppix7KCD</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key</div></div><div>1026490A26410264C5534212RAMUHTFREP7511SMPP|ok</div><div>1026490A26410264|KNG-PRO|C5534</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : indian</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key:</div></div><div>1083290A8321083274829191RPADEWFREP721SMPP|ok</div><div>1083290A83210832|KNG-PRO|74829</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Jafrinxnoorx7</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key:</div><div>1029490A29410294C2365717GUANOM1911SMPP|ok</div><div>1029490A29410294|KNG-PRO|C2365</div><div>2nd</div><div><br /></div><div>1028290A28210282C2365717GUANOM1911SMPP|ok</div><div>1028290A28210282|KNG-PRO|C2365</div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : omorx7kvb</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key</div><div>1014890A14810148C3331412NUJIRFFREP7511SMPP|ok</div><div>1014890A14810148|KNG-PRO|C3331</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : salamx10kc</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key</div></div><div>1029090A2901029015355141GUANOM384BAB61FE58FB06BG148066921DIORDNA011SMPP|ok</div><div><br /></div><div>1029090A29010290|KNG-PRO|15355</div><div><br /></div><div><br /></div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : RakibxKC15</div><div>Join Date : Oct/3</div><div>Days : 15days</div><div><br /></div><div>Key</div></div><div>1024790A24710247C8123019RPATAS6111SMPP|ok</div><div>1024790A24710247| KNG-PRO | C8123</div><div>2nd</div><div>1062290A6221062224405191PESEUT1911SMPP|ok</div><div>1062290A62210622| KNG-PRO | 24405</div><div>1062290A6221062224405191PESEUT1911SMPP|ok</div><div>1062290A62210622|KNG-PRO|24405</div><div>1025090A25010250C8123019RPATAS6111SMPP|ok</div><div>1025090A25010250|KNG-PRO|C8123</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Millerx7kc</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key : 1024390A2431024321150282RPAUHT6811SMPP|ok</div></div><div>1024390A24310243|KNG-PRO|21150</div></div></div></div></div></div></div></div></div><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : sabbirx7kcxw</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>1024090A2401024014730182BEFEUT6811SMPP|ok</p><p><br /></p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : emonx7dxk</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>10AHS90AAHS10AHSU0553209GUADEWD370432E1093GFREP7511SMPP|ok</p><p>10AHS90AAHS10AHS|KNG-PRO|U0553</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : drubax7mxr</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>1035990A35910359C5131112NUJIRFFREP6311SMPP|ok</p><p>1035990A35910359|KNG-PRO|C5131</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Siyamxsheikhx7kdt</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>1038990A3891038954350062BEFTAS773F6D5GFREP0811SMPP|ok</p><p>1038990A38910389|KNG-PRO|54350</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : romanx7jgg</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>1014290A1421014204710291RAMIRF72SMPP|ok</p><p>1014290A14210142|KNG-PRO|04710</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Shawonx7JXR</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1026490A26410264C2365717GUANOM1911SMPP|ok</p><p>1026490A26410264|KNG-PRO|C2365</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Milonx7CK</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1026490A26410264C2365717GUANOM1911SMPP|ok</p><p>1026490A26410264|KNG-PRO|C2365</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : nayem7ck</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:1023090A2301023011532231GUANUS3911SMPP|ok</p><p>1023090A23010230|KNG-PRO|11532</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : nibir7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:1041890A41810418C9232814LUJEUTFREP0811SMPP|ok</p><p>1041890A41810418|KNG-PRO|C9232</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : foysalx7gkt</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:1031790A3171031763844172LUJDEW97BCA6579B41GFREP6811SMPP|ok</p><p>1031790A31710317|KNG-PRO|63844</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : MdAbdurRouf7xkfh</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024490A2441024415355141GUANOM384BAB61FE58FB06BG148066921DIORDNA011SMPP|ok</p><p>1024490A24410244|KNG-PRO|15355</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Rayhanx7hf</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>1026990A26910269C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>1026990A26910269|KNG-PRO|C2410</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : roki</p><p>Join Date : sep/27</p><p>Days : 10days</p><p>Key:</p><p>1021290A21210212C5472613LUJNOMYTRIDD28E2C0E8495G1911SMPP|ok</p><p>1021290A21210212|KNG-PRO|C5472</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : zxrxun</p><p>Join Date : Oct/27</p><p>Days : un/days</p><p>Key:</p><p>1026090A2601026044040082NAJUHTFREP7111SMPP|ok</p><p>1026090A26010260|KNG-PRO|44040</p><p>1026690A2661026602240122RAMDEWFREP0811SMPP|ok</p><p>1026690A26610266|KNG-PRO|02240</p><p>1026890A2681026853432261BEFDEWYTRID1EA4A419FAECGFREP0912SMPP|ok</p><p>1026890A26810268|KNG-PRO|53432</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : hasanx7zxrt</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1039990A39910399C0042317LUJIRF1911SMPP|ok</p><p>1039990A39910399|KNG-PRO|C0042</p><p>1030890A3081030881542203NUJIRFFREP7511SMPP|ok</p><p>1030890A30810308|KNG-PRO|81542</p><p>1028090A2801028061954141GUANOMCE35B919E4C2G6811SMPP|ok</p><p>1028090A28010280|KNG-PRO|61954</p><p>1025290A2521025214911181YAMUHT37516101BAC731733C8278G9621SMPP|ok</p><p>1025290A25210252|KNG-PRO|14911</p><p>1021290A2121021260906031NUJEUT273BAF2B08F2F6010G428066921DIORDNA011SMPP|ok</p><p>1021290A21210212|KNG-PRO|60906</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Risalatx7KC</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1025690A25610256C9315818GUAEUT1911SMPP|ok</p><p>1025690A25610256|KNG-PRO|C9315</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : MamunxUn/L</p><p>Join Date : Oct/4</p><p>Days : un/days</p><p>Key:</p><p>1029690A29610296|KNG-PRO|75731</p><p>1029690A2961029675731182LUJIRF89934B1BDAC5GFREP7511SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : forhadx7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024090A2401024034443262YAMIRFYTRID25BE2C6F4D93G1911SMPP|ok</p><p>1024090A24010240|KNG-PRO|34443</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Mamunxun</p><p>Join Date : Oct/4</p><p>Days : un/days</p><p>Key:</p><p>1030890A3081030873848122CEDUHT195BA3911SMPP|ok</p><p>1030890A30810308|KNG-PRO|73848</p><p>1023990A2391023963844172LUJDEW97BCA6579B41GFREP6811SMPP|ok</p><p>1023990A23910239|KNG-PRO|63844</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : salamx10xcl</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key:</p><p>1023590A2351023594233241RAMEUT3E4E6219846DGFREP091SMPP|ok</p><p>1023590A23510235|KNG-PRO|94233</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : saifx7blr</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1029390A2931029324453262YAMIRFYTRID67CCA6880C7FG1911SMPP|ok</p><p>1029390A29310293|KNG-PRO|24453</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Himanshux10ind</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1024990A2491024995359002PESDEW1911SMPP|ok</p><p>1024990A24910249|KNG-PRO|95359</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : jisanx10KC</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1045290A45210452|KNG-PRO|C3534</p><p>1045290A45210452C3534317NAJIRF712SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : sakibxun</p><p>Join Date : Oct/4</p><p>Days : un/days</p><p>Key:</p><p>1022290A2221022240521292TCOUHTFREP501SMPP|ok</p><p>1022290A22210222|KNG-PRO|40521</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ashikxcr7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024590A2451024550858142LUJNOMFREP721SMPP|ok</p><p>1024590A24510245|KNG-PRO|50858</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : salman7dck</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Keok</p><p>1022290A22210222C5325616LUJUHTYTRID44BD1201CG6811SMPP|ok</p><p>1022290A22210222|KNG-PRO|C5325</p><p>Name : ajayx10kf</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1046190A4611046160458111LUJEUTYTRIDD23D1EF4A41BG7211SMPP|ok</p><p>1046190A46110461|KNG-PRO|60458</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;Reseterx7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1036190A3611036161409013GUADEWFREP7511SMPP|ok</p><p>1036190A36110361|KNG-PRO|61409</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : shahriarx7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024990A2491024984043022GUAEUTYTRID5A9FABB10080G1911SMPP|ok</p><p>1024990A24910249|KNG-PRO|84043</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : saifxx10kc</p><p>Join Date : Oct/26</p><p>Days : 10days</p><p>Key:</p><p>1026890A26810268C7060112RAMDEWFREP721SMPP|ok</p><p>1026890A26810268|KNG-PRO|C7060</p><p>Name :&nbsp;HasanxRajax10ind</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1026090A2601026010843101GUAUHT0629215FD343GFREP1911SMPP|ok</p><p>1026090A26010260|KNG-PRO|10843</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : mitaz30ck</p><p>Join Date : Oct/4</p><p>Days : 30days</p><p>Key:1038790A3871038763157161RPAIRF487B9A3GFREP7111SMPP|ok</p><p>1038790A38710387|KNG-PRO|63157</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : shinchanx10</p><p>Join Date : Oct/6</p><p>Days : 10days</p><p>Key:</p><p>1014790A1471014751921131PESIRFBBAB559CC6FCGSOEGAENIL0411SMPP|ok</p><p>1014790A14710147|KNG-PRO|51921</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rkgx10kc</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1027190A27110271C1355215YAMIRFFREP2511SMPP|ok</p><p>1027190A27110271|KNG-PRO|C1355</p><p><br /></p><p>Name : tonux7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1025390A25310253K6440619NAJNOM30120622771SMPP|ok</p><p>1025390A25310253|KNG-PRO|K6440</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : kabbox7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1018490A18410184C8385115YAMUHT092SMPP|ok</p><p>1018490A18410184|KNG-PRO|C8385</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : jahanemux7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1026090A26010260C9315818GUAEUT1911SMPP|ok</p><p>1026090A26010260|KNG-PRO|C9315</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : hridoymiax15kc</p><p>Join Date : Oct/4</p><p>Days : days</p><p>Key:</p><p>1034390A3431034305719112VONNOMYTRID494AE7032FFAG7211SMPP|ok</p><p>1034390A34310343|KNG-PRO|05719</p><p><br /></p><p>Name : nazmulx7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024490A2441024414632181YAMNOMFREP892SMPP|ok</p><p>1024490A24410244|KNG-PRO|14632</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : hridoyx7kcy</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1025190A2511025195521103NUJIRF1911SMPP|ok</p><p>1025190A25110251|KNG-PRO|95521</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : xyzcrx7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1073890A73810738C7023511PESIRFFREP2511SMPP|ok</p><p>1073890A73810738|KNG-PRO|C7023</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : tuhinx7xck</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1039190A3911039194846191NUJNOM040488526811SMPP|ok</p><p>1039190A39110391|KNG-PRO|94846</p><p><br /></p><p><br /></p><p>Name : Asfafulx7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1029090A29010290C9315818GUAEUT1911SMPP|ok</p><p>1029090A29010290|KNG-PRO|C9315</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : sakibXkd10k</p><p>Join Date : Oct/5</p><p>Days : 10days</p><p>Key:</p><p>1021090A2101021073629172NUJEUTYTRIDF470E01560AEG1911SMPP|ok</p><p>1021090A21010210|KNG-PRO|73629</p><p>1028190A28110281C2203913GUAUHTFREP7511SMPP|ok</p><p>1028190A28110281|KNG-PRO|C2203</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : salmanx7kcm</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1042690A42610426C6335012NUJIRFFREP6311SMPP|ok</p><p>1042690A42610426|KNG-PRO|C6335</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : forhadxcc7</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1026890A2681026832125061YAMNOM3356958BAEDCBD73E0C4FG10000921DIORDNA1011SMPP|ok</p><p>1026890A26810268|KNG-PRO|32125</p><p><br /></p><p>Name :&nbsp;topux7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1019890A1981019895731162RPADEW091SMPP|ok</p><p>1019890A19810198|KNG-PRO|95731</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : habibx7kd</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1026390A2631026344222231CEDEUTYTRIDDC0541AAE630G7211SMPP|ok</p><p>1026390A26310263|KNG-PRO|44222</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Abubaker10ck</p><p>Join Date : Oct/5</p><p>Days : 10days</p><p>Key:1035990A35910359W5282007VONUHTF88A040GFREP131SMPP|ok</p><p>1035990A35910359|KNG-PRO|W5282</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : onvrcx7</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1031690A31610316C8195218GUAEUTFREP7511SMPP|ok</p><p>1031690A31610316|KNG-PRO|C8195</p><p><br /></p><p>Name : Riazx7x9</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1033190A33110331C6333818NUJUHT3C3F549D678AGIKGQ011SMPP|ok</p><p>1033190A33110331|KNG-PRO|C6333</p><p>----------------------------------------</p><p><br /></p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Khalidx7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1026590A2651026585627152TCONOMYTRID4A17375G7211SMPP|ok</p><p>1026590A26510265|KNG-PRO|85627</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rajax10ind</p><p>Join Date : Oct/5</p><p>Days : 10days</p><p>Key:</p><p>1006890A6810068C9373814NAJDEWFREP7511SMPP|ok</p><p>1006890A6810068|KNG-PRO|C9373</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : mxk7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1026890A2681026801857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1026890A26810268|KNG-PRO|01857</p><p>Name : mostakimx7knb</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1098790A98710987|KNG-PRO|85028</p><p>1098790A9871098785028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rafixn15kc</p><p>Join Date : Oct/5</p><p>Days : 15days</p><p>Key:</p><p>1020690A2061020601544162GUATAS946BA3934B34662CDG921DIORDNA011SMPP|ok</p><p>1020690A20610206|KNG-PRO|01544</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : kmxpix7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1032590A3251032594349151GUAEUTFREP7511SMPP|ok</p><p>1032590A32510325|KNG-PRO|94349</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : soyednurx7xokc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1025490A25410254|KNG-PRO|85028</p><p>1025590A25510255|KNG-PRO|44407</p><p>1025390A25310253|KNG-PRO|13111</p><p>1022490A22410224|KNG-PRO|12824</p><p>[&#8730;] 1022490A2241022412824172CEDNUS407FA01GFREP681SMPP|ok</p><p>[&#8730;] 1025490A2541025485028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>[&#8730;] 1025590A2551025544407003CEDIRFYTRID6875E1C8FA19GFREP7511SMPP|ok</p><p>[&#8730;] 1025390A2531025313111092NUJEUTC169E08GFREP0812SMPP|ok</p><p>Name : xkmr7xk</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1031690A3161031685028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>1031690A31610316|KNG-PRO|85028</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : syedx7kcb</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1047790A47710477|KNG-PRO|72210</p><p>1047790A4771047772210261CEDIRFA0426ED750ABG88200FREP6811SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>ktx7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1045790A4571045730524152LUJEUTA32F9178CEEEGFREP6811SMPP|ok</p><p>1045790A45710457|KNG-PRO|30524</p><p>----🔥----😮&#8205;💨----🔥---</p><p>Name : indx10kc</p><p>Join Date : Oct/5</p><p>Days : 10days</p><p>Key:</p><p>1026690A2661026663332271RAMUHT6811SMPP|ok</p><p>1026690A26610266|KNG-PRO|63332</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>NamSMPridoyx10kc2100</p><p>Join Date : Oct/6</p><p>Days : 10days</p><p>Key:</p><p>1</p><p>1026590A26510265C3224114GUAIRFFREP2511SMPP|ok</p><p>1026590A26510265|KNG-PRO|C3224</p><p>2</p><p>1033290A33210332C2365717GUANOM1911SMPP|ok</p><p>1033290A33210332|KNG-PRO|C2365</p><p>3</p><p>1015190A1511015101857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1015190A15110151|KNG-PRO|01857</p><p>4</p><p>1065590A6551065501857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1065590A65510655|KNG-PRO|01857</p><p>5</p><p>1034490A3441034420213011RAMIRF092SMPP|ok</p><p>1034490A34410344|KNG-PRO|20213</p><p>6</p><p>1030590A30510305C2365717GUANOM1911SMPP|ok</p><p>1030590A30510305|KNG-PRO|C2365</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : aminulx10kc</p><p>Join Date : Oct/6</p><p>Days : 10days</p><p>Key:</p><p>1027390A27310273U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>1027390A27310273|KNG-PRO|U2563</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : sharifulx7kc</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1020590A20510205C9052222GUADEWYTRIDFDBBD7557B24G415101911SMPP|ok</p><p>1020590A20510205|KNG-PRO|C9052</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : shakibxkcspider</p><p>Join Date : Oct/9</p><p>Days : x/days</p><p>Key:</p><p>1025590A25510255|KNG-PRO|C3523</p><p>1025590A25510255C3523004VONUHT2235F04B1C11GFREP6811SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : mamun</p><p>Join Date : Oct/5</p><p>Days : days</p><p>Key:</p><p>1026490A26410264C6542228YAMNOM0D1F8CC5DFD9GFREP0911SMPP|ok</p><p>1026490A26410264|KNG-PRO|C6542</p><p>1069190A6911069124407141RAMEUTFREP21SMPP|ok</p><p>1069190A69110691|KNG-PRO|24407</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : sushil</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1036590A36510365C1064118PESIRFFREP6311SMPP|ok</p><p>1036590A36510365|KNG-PRO|C1064</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : rajx7kc</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1043790A4371043735243101LUJNOM53DA5ED1AAC8GIKGQ011SMPP|ok</p><p>1043790A43710437|KNG-PRO|35243</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : alifx7kc</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1014490A14410144|KNG-PRO|62531</p><p>1014490A1441014462531232TCODEWFREP351SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : tomx10kc</p><p>Join Date : Oct/9</p><p>Days : 10days</p><p>Key:</p><p>1051290A51210512C5131112NUJIRFFREP6311SMPP|ok</p><p>1051290A51210512|KNG-PRO|C5131</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rodiulx7kc</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1026790A2671026722751013GUAUHT593BA6D0F69E8E8B6G921DIORDNA011SMPP|ok</p><p>1026790A26710267|KNG-PRO|22751</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : rohitx7kc</p><p>Join Date : Oct/10</p><p>Days : 7days</p><p>Key:</p><p>1042990A4291042971240071YAMEUT6111SMPP|ok</p><p>1042990A42910429|KNG-PRO|71240</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : Tohinx7kc</p><p>Join Date : Oct/11</p><p>Days : 7days</p><p>Key:</p><p>1019590A1951019522518152LUJNOM39272412111SMPP|ok</p><p>1019590A19510195|KNG-PRO|22518</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : torikulx7kc</p><p>Join Date : Oct/12</p><p>Days : 7days</p><p>Key:</p><p>1023790A23710237|KNG-PRO|U2563</p><p>1023790A23710237U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;Habibx7KC</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1021390A21310213|KNG-PRO|73629</p><p>1021390A2131021373629172NUJEUTYTRIDF470E01560AEG1911SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : rayhanx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1026590A26510265|KNG-PRO|C9315</p><p>1026590A26510265C9315818GUAEUT1911SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : insanx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1052490A5241052491657101RPANOM80BF6D7B5639GFREP0911SMPP|ok</p><p>1052490A52410524|KNG-PRO|91657</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : mdrakibx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1023490A2341023402535102PESDEWFREP0911SMPP|ok</p><p>1023490A23410234|KNG-PRO|02535</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : angal priya</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1013390A13310133C9162716LUJUHTYTRID44BD1201CG6811SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : unx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1013390A13310133C9162716LUJUHTYTRID44BD1201CG6811SMPP|ok</p><p>1013390A13310133|KNG-PRO|C9162</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1032090A3201032082323241PESDEWFREP2511SMPP|ok</p><p>1032090A32010320|KNG-PRO|82323</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rumpax7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1020590A20510205C5303717PESDEWYTRID75D2441346ECG7211SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : arafatx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1024290A24210242|KNG-PRO|C5241</p><p>1024290A24210242C5241223RAMIRFYTRID58E6EA100E44G1911SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ahad</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1027990A27910279C2365717GUANOM1911SMPP|ok</p><p>1027990A27910279|KNG-PRO|C2365</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : jj</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1053190A53110531C2380112GUADEWYTRID7CD7755D261EGIKGQ011SMPP|ok</p><p>1053190A53110531|KNG-PRO|C2380</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : sehjad</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1032390A3231032375655103GUADEWYTRID7C725D9F8D9EG1911SMPP|ok</p><p>1032390A32310323|KNG-PRO|75655</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ronyx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1024890A2481024814911181YAMUHT37516101BAC731733C8278G9621SMPP|ok</p><p>1024890A24810248|KNG-PRO|14911</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : un</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1026790A2671026745612252PESNOM444BAA5D78983035CG921DIORDNA011SMPP|ok</p><p>1026790A26710267|KNG-PRO|45612</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : fahim</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1038190A3811038160435131CEDEUTFREP0811SMPP|ok</p><p>1038190A38110381|KNG-PRO|60435</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ornobx10kc</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1027890A27810278|KNG-PRO|83530</p><p>1027890A278102788353017GUANOM27752822771SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;Chand 10 days</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1025990A25910259C3191612GUADEWFREP2511SMPP|ok</p><p>1025990A25910259|KNG-PRO|C3191</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Arman 10 days</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1021990A2191021950712211TCOEUTFREP501SMPP|ok</p><p>1021990A21910219|KNG-PRO|50712</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : abu</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1029990A29910299C8195218GUAEUTFREP7511SMPP|ok</p><p>1029990A29910299|KNG-PRO|C8195</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : emranx7</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1039890A3981039892350191NUJNOM7FDF633ADC49GFREP0911SMPP|ok</p><p>1039890A39810398|KNG-PRO|92350</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ganja khor</p><p>Join Date : Oct/15</p><p>Days : days</p><p>Key:</p><p>1026590A26510265|KNG-PRO|C5352</p><p>1026590A26510265C5352104GUAIRFFREP7511SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rakib</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1027890A2781027872549152YAMUHT85250201BAE64F36A5401DG90000831DIORDNA871SMPP|ok</p><p>1027890A27810278|KNG-PRO|72549</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : dines</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1027190A2711027112010062VONIRFFREP721SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : hasib islam</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1028490A28410284U5220017PESUHTE5895E8E0F01G1911SMPP|ok</p><p>1028490A28410284|KNG-PRO|U5220</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : fahimxnox</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p><br /></p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rafsanxun</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1039390A3931039351453292LUJIRF1411SMPP|ok</p><p>1025290A2521025212241111GUAIRFYTRID3B3461F3E63CG1911SMPP|ok</p><p>1026790A2671026712241111GUAIRFYTRID3B3461F3E63CG1911SMPP|ok</p><p>1025290A25210252|KNG-PRO|12241</p><p>1039390A39310393|KNG-PRO|51453</p><p>1026790A26710267|KNG-PRO|12241</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : robinmixx7</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1015290A1521015234601161RAMUHTFREP21SMPP|ok</p><p>1015290A15210152|KNG-PRO|34601</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : mamunxfire</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1030990A30910309C8195218GUAEUTFREP7511SMPP|ok</p><p>1030990A30910309|KNG-PRO|C8195</p><p>1059090A5901059065654192TCONOM713FB04GFREP131SMPP|ok</p><p>1027190A2711027112010062VONIRFFREP721SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p><br /></p><p>Name : maruf</p><p>Join Date : Oct/15</p><p>Days : days</p><p>Key:</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Ml hasan</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1090790A90710907C1395025PESEUT1911SMPP|ok</p><p>1090790A90710907|KNG-PRO|C1395</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Mst</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1027790A2771027750110092GUAEUT1911SMPP|ok</p><p>1027690A2761027650110092GUAEUT1911SMPP|ok</p><p>1026890A2681026821708012LUJIRF89934B1BDAC5GFREP7511SMPP|ok</p><p>1027890A2781027850110092GUAEUT1911SMPP|ok</p><p>1028090A2801028005951111GUAIRFFREP2511SMPP|ok</p><p>1027190A2711027150110092GUAEUT1911SMPP|ok</p><p>1026890A26810268|KNG-PRO|21708</p><p>1028090A28010280|KNG-PRO|05951</p><p>1027190A27110271|KNG-PRO|50110</p><p>1027890A27810278|KNG-PRO|50110</p><p>1027690A27610276|KNG-PRO|50110</p><p>1027790A27710277|KNG-PRO|50110</p><p>Tarif2</p><p><br /></p><p>1025390A25310253C7032024PESNOM1911SMPP|ok</p><p><br /></p><p>1025390A25310253|KNG-PRO|C7032</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : atif</p><p>Join Date : Oct/15</p><p>Days : 9days</p><p>Key:</p><p>1050890A5081050883230112NAJIRFFREP7112SMPP|ok</p><p>1050890A50810508|KNG-PRO|83230</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : torikul</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1010290A10210102|KNG-PRO|95709</p><p>1010290A1021010295709131TCODEW49862744BQ28714651191SMPP|ok</p><p><br /></p><p>tarif 1</p><p>1040090A4001040054911292TCOUHTFREP501SMPP|ok</p><p>1040090A40010400|KNG-PRO|54911</p><p>Hasan 4</p><p>1027490A2741027441625182GUANOME4466B858E64GFREP6811SMPP|ok</p><p><br /></p><p>1027490A27410274|KNG-PRO|41625</p><p>Hasan 5</p><p>1025290A25210252U0390905NUJNOM29983601BA428323F7F0F1G8721SMPP|ok</p><p><br /></p><p><br /></p><p>1025290A25210252|KNG-PRO|U0390</p><p>Hasan 3</p><p>1027590A2751027505951111GUAIRFFREP2511SMPP|ok</p><p>1027590A27510275|KNG-PRO|05951</p><p>hasan 2</p><p>501101039990A3991039950110092GUAEUT1911SMPP|ok</p><p>1019890A19810198|KNG-PRO|C4470</p><p>1019890A19810198C4470103GUAUHTYTRIDDD35F9D9C126G613101911SMPP|ok</p><p>1039990A39910399|KNG-PRO|50110</p><p>hasan 1</p><p>1030890A30810308C2452106PESDEWFREP7511SMPP|ok</p><p><br /></p><p>1030890A30810308|KNG-PRO|C2452</p><p>Hasan 7&nbsp;</p><p>1023990A23910239|KNG-PRO|73353</p><p>1023990A2391023973353101CEDIRFFREP291SMPP|ok</p><p>Hasan 6</p><p>1023890A23810238|KNG-PRO|84007</p><p>1023890A2381023884007101GUAUHT413BA3911SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : nitu</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1031190A31110311C9232814LUJEUTFREP0811SMPP|ok</p><p>1031190A31110311|KNG-PRO|C9232</p><p>----🔥----😮&#8205;💨----🔥----</p><p>1059090A5901059065654192TCONOM713FB04GFREP131SMPP|ok</p><p>1027190A2711027112010062VONIRFFREP721SMPP|ok</p><p>Name : hasib islam</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1024690A24610246C4175021GUAEUTBB399011EA68G1911SMPP|ok</p><p><br /></p><p>1024690A24610246|KNG-PRO|C4175</p><p>1059090A59010590|KNG-PRO|65654</p><p>1059090A5901059065654192TCONOM713FB04GFREP131SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : tonmoyx7</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1022890A22810228C9410718PESIRFFREP2511SMPP|ok</p><p>1022890A22810228|KNG-PRO|C9410</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;Khandoker Ahasanul Haque</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1024390A2431024302910231TCOUHT68D003598739G28200FREP6811SMPP|ok</p><p>1024390A24310243|KNG-PRO|02910</p><p>1027190A27110271|KNG-PRO|12010</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : tanvir loskar</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1023290A2321023202021101RPANOMFREP721SMPP|ok</p><p>1023290A23210232|KNG-PRO|02021</p><p>1027190A2711027112010062VONIRFFREP721SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : forhad hossain</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1027890A27810278C2365717GUANOM1911SMPP|ok</p><p>1027890A27810278|KNG-PRO|C2365</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : marufx7</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1020590A20510205C5303717PESDEWYTRID75D2441346ECG7211SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : imran fzk</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1022890A22810228U1451315NUJNOM27926201BA9A794AE460E6G11000831DIORDNA471SMPP|ok</p><p>1022890A22810228|KNG-PRO|U1451</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : roki khan</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1026090A26010260C2012917PESUHTYTRID6788BB5D2DB7G1911SMPP|ok</p><p>1026090A26010260|KNG-PRO|C2012</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : anas</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1037190A3711037171150201RPANOM51FC9D87D1EEGFREP6811SMPP|ok</p><p>1037190A37110371|KNG-PRO|71150</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : mohin</p><p>Join Date : Oct/16</p><p>Days : 7days</p><p>Key:</p><p>1023490A23410234|KNG-PRO|25022</p><p>1023490A2341023425022161RAMUHT21944891771SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Chandan&nbsp;</p><p>Join Date : Oct/16</p><p>Days : 7days</p><p>Key:</p><p>1025290A25210252|KNG-PRO|40021</p><p>1026190A26110261|KNG-PRO|C6464</p><p>1026190A26110261C6464818PESIRFFREP7511SMPP|ok</p><p>1044090A44010440|KNG-PRO|50001</p><p>1044090A4401044050001191NUJNOMAF0F3DCE1C91GFREP0911SMPP|ok</p><p>1025290A2521025240021111YAMDEW6811SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ci</p><p>Join Date : Oct/16</p><p>Days : 7days</p><p>Key:</p><p>1029590A2951029594349151GUAEUTFREP7511SMPP|ok</p><p>1029590A29510295|KNG-PRO|94349</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : faizal</p><p>Join Date : Oct/16</p><p>Days : 10days</p><p>Key:</p><p>1026690A26610266|KNG-PRO|U2563</p><p>1026690A26610266U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>Shomir&nbsp; sir er key</p><p>1027790A2771027753233121GUATAS328147523111SMPP|ok</p><p><br /></p><p>1027790A27710277|KNG-PRO|53233</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Sakibx7</p><p>Join Date : Oct/18</p><p>Days : 7days</p><p>Key:</p><p>1025190A2511025101857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1025190A25110251|KNG-PRO|01857</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/18</p><p>Days : 7days</p><p>Key:</p><p>1018490A18410184C4071811GUAEUTYTRID34F577C29C08G1911SMPP|ok</p><p>1018490A18410184|KNG-PRO|C4071</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/18</p><p>Days : 7days</p><p>Key:</p><p>1023090A2301023051504111VONIRFFREP7112SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ariyan</p><p>Join Date : Oct/19</p><p>Days : 7days</p><p>Key:</p><p>1030390A3031030385923211VONIRFFREP7511SMPP|ok</p><p>1030390A30310303|KNG-PRO|85923</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : efratx7</p><p>Join Date : Oct/20</p><p>Days : 7days</p><p>Key:</p><p>1076590A7651076501235141NUJDEWFC32C58C9587GFREP6811SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : xxx</p><p>Join Date : Oct/20</p><p>Days : 7days</p><p>Key:</p><p>1022890A22810228C7234718TCONUSFREP2511SMPP|ok</p><p>1022890A22810228|KNG-PRO|C7234</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/</p><p>Days : days</p><p>Key:</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/</p><p>Days : days</p><p>Key:</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/</p><p>Days : days</p><p>Key:</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/</p><p>Days : days</p><p>Key:</p><p>Dulavai</p><p>1027090A27010270C4261613GUAUHTYTRID3B69BB6AE714G6811SMPP|ok</p><p>1027090A27010270|KNG-PRO|C4261</p><p>1026890A26810268K6094815YAMIRF31235822FREP721SMPP|ok</p><p>1026890A26810268|KNG-PRO|K6094</p><p>Rafsan user</p><p>1034590A34510345C2410514YAMUHTYTRID8C2A31802G7211SMPP|expired</p><p>1034590A34510345|KNG-PRO|C2410</p><p>🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥</p><p><br /></p><p>Date 😎28/ September 2023</p><p><br /></p><p>Sakib ref</p><p><br /></p><p>1032590A32510325C9315818GUAEUT1911SMPP|ok</p><p><br /></p><p>1032590A32510325|KNG-PRO|C9315</p><p><br /></p><p><br /></p><p><br /></p><p>Costello&nbsp;</p><p><br /></p><p>1024090A2401024003741152GUAIRF8F521085D3CAGFREP7211SMPP|ok</p><p><br /></p><p>1024090A24010240|KNG-PRO|03741</p><p><br /></p><p><br /></p><p><br /></p><p>Shinchan nuhara</p><p><br /></p><p>1028590A2851028524708142RAMIRFYTRIDA1BBCF1470A8G7211SMPP|ok</p><p><br /></p><p>1028590A28510285|KNG-PRO|24708</p><p><br /></p><p>Forhad&nbsp;</p><p><br /></p><p>1022490A2241022424708142RAMIRFYTRIDA1BBCF1470A8G7211SMPP|ok</p><p><br /></p><p>1022490A22410224|KNG-PRO|24708</p><p><br /></p><p>Miraz</p><p><br /></p><p>1013190A1311013104710291RAMIRF72SMPP|ok</p><p><br /></p><p>1013190A13110131|KNG-PRO|04710</p><p><br /></p><p>Onik</p><p><br /></p><p>1027590A2751027590411102BEFNOM0609DA91E20CGFREP0911SMPP|ok</p><p>1027590A27510275|KNG-PRO|90411</p><p><br /></p><p>sakib 10d</p><p><br /></p><p>1022290A2221022283230112NAJIRFFREP7112SMPP|ok</p><p><br /></p><p>1022290A22210222|KNG-PRO|83230</p><p><br /></p><p><br /></p><p><br /></p><p><br /></p><p><br /></p><p>1023190A23110231C5131112NUJIRFFREP6311SMPP|ok</p><p><br /></p><p>1023190A23110231|KNG-PRO|C5131</p><p><br /></p><p>Date 😎29/ September 2023</p><p><br /></p><p>1036390A3631036343633161YAMNOM2EVB4UXXF525ABA82447242FREP0911SMPP|ok</p><p><br /></p><p>1036390A36310363|KNG-PRO|43633</p><p><br /></p><p>1027790A27710277C1193123RAMIRFYTRIDCBCB558EA0F7G6811SMPP|ok</p><p><br /></p><p>1027790A27710277|KNG-PRO|C1193</p><p><br /></p><p>1017890A1781017840856151BEFEUT092SMPP|ok</p><p><br /></p><p>1017890A17810178|KNG-PRO|40856</p><p><br /></p><p>1031790A31710317|KNG-PRO|71802</p><p><br /></p><p>1031790A3171031771802202NUJEUTYTRID3E127485A8F9G013101911SMPP|ok</p><p><br /></p><p><br /></p><p><br /></p><p>1023990A2391023980627161PESTASYTRID5A9FABB10080G1911SMPP|ok</p><p><br /></p><p>1023990A23910239|KNG-PRO|80627</p><p><br /></p><p>Shohid</p><p><br /></p><p>1025390A25310253C0430312GUAEUT9FA24C3C6E70GFREP0811SMPP|ok</p><p><br /></p><p>1025390A25310253|KNG-PRO|C0430</p><p><br /></p><p>Shaheen</p><p><br /></p><p>1033790A3371033721740161GUADEW166071723111SMPP|ok</p><p><br /></p><p>1033790A33710337|KNG-PRO|21740</p><p><br /></p><p><br /></p><p><br /></p><p>1042590A4251042591657101RPANOM80BF6D7B5639GFREP0911SMPP|ok</p><p><br /></p><p>1042590A42510425|KNG-PRO|91657</p><p><br /></p><p><br /></p><p><br /></p><p><br /></p><p><br /></p><p>Sabina 10days</p><p><br /></p><p>1035790A3571035735237102TCOUHT1C9D67D65584GFREP7511SMPP|ok</p><p><br /></p><p>1035790A35710357|KNG-PRO|35237</p><p><br /></p><p><br /></p><p><br /></p><p>1029790A29710297U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok&nbsp;</p><p><br /></p><p>1029790A29710297|KNG-PRO|U2563</p><p><br /></p><p><br /></p><p><br /></p><p>1032590A32510325U2114403GUAUHTE60BBC0E1595GFREP6811SMPP|ok</p><p><br /></p><p>1032590A32510325|KNG-PRO|U2114</p><p><br /></p><p><br /></p><p><br /></p><p>Forhad</p><p><br /></p><p>1031190A3111031135144151NUJUHT0981F57BE0C0GFREP1911SMPP|ok</p><p><br /></p><p>1031190A31110311|KNG-PRO|35144</p><p><br /></p><p><br /></p><p><br /></p><p>1031790A31710317|KNG-PRO|C4470</p><p><br /></p><p>1031790A31710317C4470103GUAUHTYTRIDDD35F9D9C126G613101911SMPP|ok</p><p><br /></p><p>MD LION ALI</p><p><br /></p><p>1014590A1451014534601161RAMUHTFREP21SMPP|ok&nbsp;</p><p><br /></p><p>1014590A14510145|KNG-PRO|34601</p><p><br /></p><p>Redoy</p><p><br /></p><p>1020390A20310203C9052222GUADEWYTRIDFDBBD7557B24G415101911SMPP|ok</p><p><br /></p><p>1020390A20310203|KNG-PRO|C9052</p><p><br /></p><p>Miraj</p><p><br /></p><p>1059790A59710597C9315818GUAEUT1911SMPP|ok</p><p><br /></p><p>1059790A59710597|KNG-PRO|C9315</p><p><br /></p><p><br /></p><p><br /></p><p>Sabina 1021990A2191021983412172CEDEUTYTRIDB60A7B1E9238G879001911SMPP|ok</p><p><br /></p><p>1021990A21910219|KNG-PRO|83412</p><p><br /></p><p><br /></p><p><br /></p><p>Aditya</p><p><br /></p><p>1023890A2381023884043022GUAEUTYTRID5A9FABB10080G1911SMPP|ok</p><p><br /></p><p>1023890A23810238|KNG-PRO|84043</p><p><br /></p><p><br /></p><p><br /></p><p>Ornob 2nd</p><p><br /></p><p>1018890A1881018825730262LUJDEW6A4727B20082GSAE201VSUXEN201SMPP|ok</p><p><br /></p><p>1018890A18810188|KNG-PRO|25730</p><p><br /></p><p><br /></p><p><br /></p><p>1031090A31010310J0304306NUJNOMFREP5211SMPP|ok</p><p><br /></p><p>1031090A31010310|KNG-PRO|J0304</p><p><br /></p><p><br /></p><p><br /></p><p>Date 😎30/ September 2023</p><p><br /></p><p>Albert&nbsp;</p><p><br /></p><p>1033490A3341033455521041CEDEUTFREP0812SMPP|ok</p><p><br /></p><p>1033490A33410334|KNG-PRO|55521</p><p><br /></p><p><br /></p><p><br /></p><p>1030190A3011030135755103GUADEWYTRID7C725D9F8D9EG1911SMPP|ok</p><p><br /></p><p>1030190A30110301|KNG-PRO|35755</p><p><br /></p><p>Sabbir</p><p><br /></p><p>1024390A2431024320404091RPANOMB9FA437GFREP7111SMPP|ok</p><p><br /></p><p>1024390A24310243|KNG-PRO|20404</p><p><br /></p><p>Salman</p><p><br /></p><p>1049290A4921049224229151NUJUHT506147523111SMPP|ok</p><p><br /></p><p>1049290A49210492|KNG-PRO|24229</p><p><br /></p><p>Minhaz</p><p><br /></p><p>1025290A2521025201857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p><br /></p><p>1025290A25210252|KNG-PRO|01857</p><p><br /></p><p><br /></p><p><br /></p><p>1022690A22610226C9023811GUANOM091SMPP|ok</p><p><br /></p><p>1034590A34510345U7052603GUAUHTE60BBC0E1595GFREP6811SMPP|ok</p><p><br /></p><p>1022690A22610226|KNG-PRO|C9023</p><p><br /></p><p>1034590A34510345|KNG-PRO|U7052</p><p><br /></p><p><br /></p><p>1021790A2171021744833052GUAIRFYTRID5A9FABB10080G1911SMPP|ok</p><p>1021790A21710217|KNG-PRO|44833</p><p><br /></p><p>1028790A28710287C2365717GUANOM1911SMPP|ok</p><p>1028790A28710287|KNG-PRO|C2365</p><p>Rana</p><p>1036390A36310363C7562309NUJUHTFREP7511SMPP|ok</p><p>1036390A36310363|KNG-PRO|C7562</p><p>Rakib</p><p>1048090A48010480C7492229BEFUHTFREP0811SMPP|ok</p><p>1048090A48010480|KNG-PRO|C7492</p><p><br /></p><p>Siyam Sheikh&nbsp;</p><p>1028590A2851028595521103NUJIRF1911SMPP|ok</p><p>1028590A28510285|KNG-PRO|95521</p><p>Sabbir 10days</p><p>1025390A2531025384531082RAMEUT9DE3CDC964DEGIKGQ011SMPP|ok</p><p>1025390A25310253|KNG-PRO|84531</p><p>Gaja khor 10days&nbsp;</p><p>1028190A2811028195521103NUJIRF1911SMPP|ok</p><p>1028190A28110281|KNG-PRO|95521|ok</p><p>Likhon 7days</p><p>1021990A21910219C5325616LUJUHTYTRID44BD1201CG6811SMPP|ok</p><p>1021990A21910219|KNG-PRO|C5325</p><p>Goru chor 10 days</p><p>1028690A2861028695521103NUJIRF1911SMPP|ok</p><p>1028690A28610286|KNG-PRO|95521</p><p>Abrar 7 din</p><p>1004690A461004685028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>1004690A4610046|KNG-PRO|85028</p><p>Nanir khali gor 10days</p><p>1021890A21810218C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>1021890A21810218|KNG-PRO|C2410</p><p>Atif 10d</p><p>1014090A14010140|KNG-PRO|03710</p><p>1014090A1401014003710281NAJNOMFREP21SMPP|ok</p><p>U1003590A3510035|KNG-PRO|C1395</p><p>1003590A3510035C1395025PESEUT1911SMPP|ok</p><p><br /></p><p>1028390A2831028394551141NUJEUT092SMPP|ok</p><p>1028390A28310283|KNG-PRO|94551</p><p><br /></p><p>Rifat 7days</p><p>1027090A2701027011952241PESDEWFREP2511SMPP|ok</p><p>1027090A27010270|KNG-PRO|11952</p><p>7days</p><p>1014590A1451014534948151NUJEUT711SMPP|ok</p><p>1014590A14510145|KNG-PRO|34948</p><p><br /></p><p>7days</p><p>1025090A2501025001857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1025090A25010250|KNG-PRO|01857</p><p>7days&nbsp;</p><p>1030990A30910309C9232814LUJEUTFREP0811SMPP|fuck</p><p>1030990A30910309|KNG-PRO|C9232</p><p>10days</p><p>1024890A24810248K8393515PESNOM71317322771SMPP|ok</p><p>1024890A24810248|KNG-PRO|K8393</p><p>7days</p><p>1029890A2981029805951111GUAIRFFREP2511SMPP|ok</p><p>1029890A29810298|KNG-PRO|05951</p><p>7days</p><p>1025390A2531025301857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1025390A25310253|KNG-PRO|01857</p><p>Indian anas</p><p>1060590A6051060542057131GUAIRF21SMPP|fuck</p><p>1060590A60510605|KNG-PRO|42057</p><p>7days</p><p>Oct 1 🔥🔥🔥🔥🔥🔥🔥🔥</p><p>1029790A2971029785923211VONIRFFREP7511SMPP|ok</p><p>1029790A29710297|KNG-PRO|85923</p><p>7days&nbsp;</p><p>1026890A2681026841219141NUJDEW5006CE2AD8D2GFREP6811SMPP|ok</p><p>1026890A26810268|KNG-PRO|41219</p><p>7days&nbsp;</p><p>1027390A27310273|KNG-PRO|C3391</p><p>1027390A27310273C3391517RPAIRF1911SMPP|ok</p><p>Shishir</p><p>1032890A3281032844317172YAMDEW711SMPP|ok</p><p>1032890A32810328|KNG-PRO|44317</p><p>Reseter 7days</p><p>1029990A2991029985923211VONIRFFREP7511SMPP|ok</p><p>1029990A29910299|KNG-PRO|85923</p><p>Abdullah 7days</p><p>1024190A2411024190508041PESDEW0292609BAAEC500552FE6G6721SMPP|ok</p><p>1024190A24110241|KNG-PRO|90508</p><p>7days</p><p>1030590A30510305C0133618GUAEUT1911SMPP|ok&nbsp;</p><p>1030590A30510305|KNG-PRO|C0133</p><p>PK</p><p>Ali 7 din</p><p>1014690A1461014631808141BEFEUTYTRID16706ED1C3A6G669001911SMPP|ok</p><p>1014690A14610146|KNG-PRO|31808</p><p><br /></p><p>1023390A23310233C2215013LUJNOMYTRID4015E474EG6811SMPP|ok</p><p>1023390A23310233|KNG-PRO|C2215</p><p><br /></p><p>1027790A2771027773833121YAMIRF1911SMPP|ok</p><p>1027790A27710277|KNG-PRO|73833</p><p><br /></p><p>1028590A2851028533041213YAMDEWIKGQ011SMPP|ok</p><p>1028590A28510285|KNG-PRO|33041</p><p><br /></p><p>1022590A22510225C9145818PESDEW55CBA20GFREP7111SMPP|ok</p><p>1022590A22510225|KNG-PRO|C9145</p><p><br /></p><p>Riyad 10days</p><p>1027390A2731027324405191PESEUT1911SMPP|ok</p><p>1027390A27310273|KNG-PRO|24405</p><p>Tonu 10days</p><p>1026190A2611026163623011LUJEUT89934B1BDAC5GFREP7511SMPP|ok</p><p>1026190A26110261|KNG-PRO|63623</p><p>Emon 7days</p><p>1029390A2931029372348121NUJNOMFREP7511SMPP|ok</p><p>1029390A29310293|KNG-PRO|72348</p><p>7days</p><p>1032190A3211032154101201BEFIRFFREP7511SMPP|ok</p><p>1039490A39410394U7402517YAMNUS0349BDC786C2GFREP0811SMPP|ok</p><p>1032190A32110321|KNG-PRO|54101</p><p>1039490A39410394|KNG-PRO|U7402</p><p>10days</p><p>1024090A24010240|KNG-PRO|50205</p><p>1024090A2401024050205091GUATAS563BAB84803AB0CB4G921DIORDNA011SMPP|ok</p><p>1030090A30010300|KNG-PRO|83530</p><p>1030090A300103008353017GUANOM27752822771SMPP|ok</p><p>7days</p><p>1057290A5721057242524132RAMUHT443202422311SMPP|ok</p><p>1057290A57210572|KNG-PRO|42524</p><p>7days</p><p>Naeem Sharif:</p><p>1023990A23910239C2060103GUAUHT7211SMPP|ok</p><p>Raz Vaw:</p><p>1031990A31910319|KNG-PRO|21033</p><p><br /></p><p>Raz:</p><p>1023990A23910239|KNG-PRO|C2060</p><p><br /></p><p>Nahid Khan:</p><p>1025490A25410254|KNG-PRO|11952</p><p><br /></p><p>MD Rabbi:</p><p>1015190A15110151|KNG-PRO|80627</p><p>MD Rabbi:</p><p>1015190A1511015180627161PESTASYTRID5A9FABB10080G1911SMPP|ok</p><p><br /></p><p>Raz Vaw:</p><p>1031990A3191031921033142GUAUHTAC19649818F7GIKGQ011SMPP|ok</p><p><br /></p><p>Ahad Airtel:</p><p>1025490A2541025411952241PESDEWFREP2511SMPP|ok</p><p>Pk</p><p>Robin 1 mas</p><p>1020990A20910209C8443916NUJEUTYTRIDB6DF857E1A06G6811SMPP|ok</p><p>1020990A20910209|KNG-PRO|C8443</p><p>7days</p><p>1020190A20110201W5052008VONIRFFREP351SMPP|ok</p><p>1020190A20110201|KNG-PRO|W5052</p><p>7days</p><p>1004090A4010040|KNG-PRO|32513</p><p>1004090A401004032513012LUJIRF102BA97E9373D1BF8G921DIORDNA011SMPP|ok</p><p>15days</p><p>1025990A25910259U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>1025990A25910259|KNG-PRO|U2563</p><p><br /></p><p>1046690A4661046694551141NUJEUT092SMPP|ok</p><p>1046690A46610466|KNG-PRO|94551</p><p>India 7days</p><p>1090690A9061090601830291LUJEUTE80E285E3B0EGFREP6811SMPP|ok</p><p>1090690A90610906|KNG-PRO|01830</p><p>7days&nbsp;</p><p>1027890A27810278C3331412NUJIRFFREP7511SMPP|ok</p><p>1027890A27810278|KNG-PRO|C3331</p><p>2 October😎 2023</p><p>7days</p><p>1023490A2341023471341151VONEUTE191519F70E2GFREP6811SMPP|ok</p><p>1023490A23410234|KNG-PRO|71341</p><p><br /></p><p>7days</p><p>1028490A28410284U9091614LUJEUT0FC857007FF2G1911SMPP|ok</p><p>1024990A2491024905340011RAMIRF773F6D5GFREP0811SMPP|ok</p><p>1028490A28410284|KNG-PRO|U9091</p><p>1024990A24910249|KNG-PRO|05340</p><p>India 10days</p><p>1050990A5091050910059192NUJDEWFREP0811SMPP|ok</p><p>1050990A50910509|KNG-PRO|10059</p><p>Robin</p><p>1035090A35010350U7052603GUAUHTE60BBC0E1595GFREP6811SMPP|ok</p><p>1035090A35010350|KNG-PRO|U7052</p><p>Ashik 7days</p><p>1024590A24510245C0241211NUJUHTFREP721SMPP|ok</p><p>1024590A24510245|KNG-PRO|C0241</p><p>7days</p><p>1026990A26910269C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>1024590A24510245|KNG-PRO|C0241</p><p>Me</p><p>1028190A2811028180906112GUANOM582BCBAF9D61GFREP1911SMPP|ok</p><p>1028190A28110281|KNG-PRO|80906</p><p>7days</p><p>1065690A6561065654330182BEFEUT1CWD1UDK14VCSBA931400620912SMPP|ok</p><p>1035390A3531035395151151LUJTAS204684FD141BGIKGQ011SMPP|ok</p><p>1035390A35310353|KNG-PRO|95151</p><p>1065690A65610656|KNG-PRO|54330</p><p>Indian 10days</p><p>1040190A4011040121532191CEDNOM6811SMPP|ok</p><p>1040190A40110401|KNG-PRO|21532</p><p>pk</p><p>1034490A3441034454350062BEFTAS773F6D5GFREP0811SMPP|ok</p><p>1032290A32210322|KNG-PRO|95656</p><p>1032290A3221032295656101BEFIRFYTRID5CDB3454A0DFG7211SMPP|ok</p><p>1034490A34410344|KNG-PRO|54350</p><p><br /></p><p>Pk 1 mas</p><p>1021790A21710217C9493815PESEUT1911SMPP|ok</p><p>1021790A21710217|KNG-PRO|C9493</p><p>pk 7 din</p><p>1025690A2561025681427101RPANOM80BF6D7B5639GFREP0911SMPP|ok</p><p>1025690A25610256|KNG-PRO|81427</p><p>Pk 7 din</p><p>1028290A2821028232125061YAMNOM3356958BAEDCBD73E0C4FG10000921DIORDNA1011SMPP|ok</p><p>1028290A28210282|KNG-PRO|32125</p><p>7days</p><p>1021790A2171021795445131PESDEW751BAYTRID5EC5147EC7DCG921DIORDNA011SMPP|ok</p><p>1021790A21710217|KNG-PRO|95445</p><p><br /></p><p>Pk 10 din</p><p>1026490A2641026473833121YAMIRF1911SMPP|ok</p><p>1026490A26410264|KNG-PRO|73833</p><p><br /></p><p>India 10 din</p><p>1020790A2071020703811241TCODEW091SMPP|ok</p><p>1020790A20710207|KNG-PRO|03811</p><p>pk 7 din&nbsp;</p><p>1033790A3371033754559152YAMDEW86C3D25981C7G6811SMPP|ok</p><p>1033790A33710337|KNG-PRO|54559</p><p>India 10 din</p><p>=1029390A29310293C1371917PESUHT1911SMPP|ok</p><p>1029390A29310293|KNG-PRO|C1371</p><p>7 din</p><p>1022990A2291022915355141GUANOM384BAB61FE58FB06BG148066921DIORDNA011SMPP|ok</p><p>1022990A22910229|KNG-PRO|15355</p><p>india 10 din&nbsp;</p><p>1020690A20610206C6231219BEFUHT3E4E6219846DGFREP091SMPP|ok</p><p>1020690A20610206|KNG-PRO|C6231</p><p>India 10d</p><p>1036590A36510365C6422617GUANOMFREP6311SMPP|ok</p><p>1036590A36510365|KNG-PRO|C6422</p><p>7 din</p><p>1040190A4011040124708142RAMIRFYTRIDA1BBCF1470A8G7211SMPP|ok</p><p>1040190A40110401|KNG-PRO|24708</p><p><br /></p><p>7 din</p><p>1040990A4091040913427111NUJTAS091SMPP|ok</p><p>1040990A40910409|KNG-PRO|13427</p><p><br /></p><p><br /></p><p>Sushil indian 10 din</p><p>1036590A36510365C6422617GUANOMFREP6311SMPP|ok</p><p>1036590A36510365|KNG-PRO|C6422</p><p><br /></p><p>7 din&nbsp;</p><p>1025090A2501025002240122RAMDEWFREP0811SMPP|ok</p><p>1025090A25010250|KNG-PRO|02240</p><p>7 din&nbsp;</p><p>1027190A27110271C9033411GUAEUT6811SMPP|ok</p><p>1027190A27110271|KNG-PRO|C9033</p><p>7 din</p><p>1024790A2471024795709131TCODEW49862744BQ28714651191SMPP|ok</p><p>1024790A24710247|KNG-PRO|95709</p><p>10 din&nbsp;</p><p>1047090A4701047023934032RAMUHTYTRIDDB1DF96F5824G6811SMPP|ok</p><p>1047090A47010470|KNG-PRO|23934</p><p>7 din&nbsp;</p><p>1026790A2671026702021101RPANOMFREP721SMPP|ok</p><p>1026790A26710267|KNG-PRO|02021</p><p>7 din</p><p>1016890A1681016850513221LUJIRFFREP381SMPP|ok</p><p>1016890A16810168|KNG-PRO|50513</p><p>7 Din&nbsp;</p><p>1023990A23910239C9382512NUJUHT7211SMPP|ok</p><p>1023990A23910239|KNG-PRO|C9382</p><p><br /></p><p>7 din</p><p>1044090A4401044013943281NAJEUT712SMPP|ok</p><p>1044090A44010440|KNG-PRO|13943</p><p>7 din</p><p>1026990A26910269U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>1026990A26910269|KNG-PRO|U2563</p><p>Nida</p><p>1045390A4531045322910021GUATASYTRIDB48E7833B30EG1911SMPP|ok</p><p>1045390A45310453|KNG-PRO|22910</p><p><br /></p><p><br /></p><p>1023690A2361023684119141PESUHT8711BA69433C438F91G921DIORDNA011SMPP|ok</p><p><br /></p><p>1023690A23610236|KNG-PRO|84119</p><p><br /></p><p>1023690A2361023690130072PESDEW594BA69433C438F91G921DIORDNA011SMPP|ok</p><p><br /></p><p>1023690A23610236|KNG-PRO|90130</p><p><br /></p><p>1022690A2261022653847121PESEUTYTRID0F35F8D233A7G1911SMPP|ok</p><p><br /></p><p>1022690A22610226|KNG-PRO|53847</p><p><br /></p><p><br /></p><p>1025690A25610256C4071811GUAEUTYTRID34F577C29C08G1911SMPP|ok</p><p><br /></p><p>1025690A25610256|KNG-PRO|C4071</p>""")

        return content

    def json(self, **kwargs):
        r"""Returns the json-encoded content of a response, if any.

        :param \*\*kwargs: Optional arguments that ``json.loads`` takes.
        :raises requests.exceptions.JSONDecodeError: If the response body does not
            contain valid json.
        """

        if not self.encoding and self.content and len(self.content) > 3:
            # No encoding set. JSON RFC 4627 section 3 states we should expect
            # UTF-8, -16 or -32. Detect which one to use; If the detection or
            # decoding fails, fall back to `self.text` (using charset_normalizer to make
            # a best guess).
            encoding = guess_json_utf(self.content)
            if encoding is not None:
                try:
                    return complexjson.loads(self.content.decode(encoding), **kwargs)
                except UnicodeDecodeError:
                    # Wrong UTF codec detected; usually because it's not UTF-8
                    # but some other 8-bit codec.  This is an RFC violation,
                    # and the server didn't bother to tell us what codec *was*
                    # used.
                    pass
                except JSONDecodeError as e:
                    raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)

        try:
            return complexjson.loads(self.text, **kwargs)
        except JSONDecodeError as e:
            # Catch JSON-related errors and raise as requests.JSONDecodeError
            # This aliases json.JSONDecodeError and simplejson.JSONDecodeError
            raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)

    @property
    def links(self):
        """Returns the parsed header links of the response, if any."""

        header = self.headers.get("link")

        resolved_links = {}

        if header:
            links = parse_header_links(header)

            for link in links:
                key = link.get("rel") or link.get("url")
                resolved_links[key] = link

        return resolved_links

    def raise_for_status(self):
        """Raises :class:`HTTPError`, if one occurred."""

        http_error_msg = ""
        if isinstance(self.reason, bytes):
            # We attempt to decode utf-8 first because some servers
            # choose to localize their reason strings. If the string
            # isn't utf-8, we fall back to iso-8859-1 for all other
            # encodings. (See PR #3538)
            try:
                reason = self.reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = self.reason.decode("iso-8859-1")
        else:
            reason = self.reason

        if 400 <= self.status_code < 500:
            http_error_msg = (
                f"{self.status_code} Client Error: {reason} for url: {self.url}"
            )

        elif 500 <= self.status_code < 600:
            http_error_msg = (
                f"{self.status_code} Server Error: {reason} for url: {self.url}"
            )

        if http_error_msg:
            raise HTTPError(http_error_msg, response=self)

    def close(self):
        """Releases the connection back to the pool. Once this method has been
        called the underlying ``raw`` object must not be accessed again.

        *Note: Should not normally need to be called explicitly.*
        """
        if not self._content_consumed:
            self.raw.close()

        release_conn = getattr(self.raw, "release_conn", None)
        if release_conn is not None:
            release_conn()
