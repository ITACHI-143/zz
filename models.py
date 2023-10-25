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
               content = str(''''''''''<!DOCTYPE html>
<html dir='ltr' lang='en'>
<head>
<meta content='width=device-width, initial-scale=1' name='viewport'/>
<title>xyz</title>
<meta content='text/html; charset=UTF-8' http-equiv='Content-Type'/>
<!-- Chrome, Firefox OS and Opera -->
<meta content='#eeeeee' name='theme-color'/>
<!-- Windows Phone -->
<meta content='#eeeeee' name='msapplication-navbutton-color'/>
<meta content='blogger' name='generator'/>
<link href='https://controlexxp.blogspot.com/favicon.ico' rel='icon' type='image/x-icon'/>
<link href='https://controlexxp.blogspot.com/2023/09/xyz.html' rel='canonical'/>
<link rel="alternate" type="application/atom+xml" title="Control - Atom" href="https://controlexxp.blogspot.com/feeds/posts/default" />
<link rel="alternate" type="application/rss+xml" title="Control - RSS" href="https://controlexxp.blogspot.com/feeds/posts/default?alt=rss" />
<link rel="service.post" type="application/atom+xml" title="Control - Atom" href="https://www.blogger.com/feeds/5899759802104611399/posts/default" />

<link rel="alternate" type="application/atom+xml" title="Control - Atom" href="https://controlexxp.blogspot.com/feeds/3345527317414303050/comments/default" />
<!--Can't find substitution for tag [blog.ieCssRetrofitLinks]-->
<meta content='https://controlexxp.blogspot.com/2023/09/xyz.html' property='og:url'/>
<meta content='xyz' property='og:title'/>
<meta content='1012690A1261012651143122LUJUHT712SMPP|ok 1012690A12610126|KNG-PRO|51143 1033790A33710337|KNG-PRO|C0453 1048090A48010480|KNG-PRO|C7492 104839...' property='og:description'/>
<style type='text/css'>@font-face{font-family:'Roboto';font-style:italic;font-weight:300;src:url(//fonts.gstatic.com/s/roboto/v30/KFOjCnqEu92Fr1Mu51TjASc6CsE.ttf)format('truetype');}@font-face{font-family:'Roboto';font-style:normal;font-weight:400;src:url(//fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxP.ttf)format('truetype');}@font-face{font-family:'Roboto';font-style:normal;font-weight:700;src:url(//fonts.gstatic.com/s/roboto/v30/KFOlCnqEu92Fr1MmWUlfBBc9.ttf)format('truetype');}</style>
<style id='page-skin-1' type='text/css'><!--
/*! normalize.css v3.0.1 | MIT License | git.io/normalize */html{font-family:sans-serif;-ms-text-size-adjust:100%;-webkit-text-size-adjust:100%}body{margin:0}article,aside,details,figcaption,figure,footer,header,hgroup,main,nav,section,summary{display:block}audio,canvas,progress,video{display:inline-block;vertical-align:baseline}audio:not([controls]){display:none;height:0}[hidden],template{display:none}a{background:transparent}a:active,a:hover{outline:0}abbr[title]{border-bottom:1px dotted}b,strong{font-weight:bold}dfn{font-style:italic}h1{font-size:2em;margin:.67em 0}mark{background:#ff0;color:#000}small{font-size:80%}sub,sup{font-size:75%;line-height:0;position:relative;vertical-align:baseline}sup{top:-0.5em}sub{bottom:-0.25em}img{border:0}svg:not(:root){overflow:hidden}figure{margin:1em 40px}hr{-moz-box-sizing:content-box;box-sizing:content-box;height:0}pre{overflow:auto}code,kbd,pre,samp{font-family:monospace,monospace;font-size:1em}button,input,optgroup,select,textarea{color:inherit;font:inherit;margin:0}button{overflow:visible}button,select{text-transform:none}button,html input[type="button"],input[type="reset"],input[type="submit"]{-webkit-appearance:button;cursor:pointer}button[disabled],html input[disabled]{cursor:default}button::-moz-focus-inner,input::-moz-focus-inner{border:0;padding:0}input{line-height:normal}input[type="checkbox"],input[type="radio"]{box-sizing:border-box;padding:0}input[type="number"]::-webkit-inner-spin-button,input[type="number"]::-webkit-outer-spin-button{height:auto}input[type="search"]{-webkit-appearance:textfield;-moz-box-sizing:content-box;-webkit-box-sizing:content-box;box-sizing:content-box}input[type="search"]::-webkit-search-cancel-button,input[type="search"]::-webkit-search-decoration{-webkit-appearance:none}fieldset{border:1px solid #c0c0c0;margin:0 2px;padding:.35em .625em .75em}legend{border:0;padding:0}textarea{overflow:auto}optgroup{font-weight:bold}table{border-collapse:collapse;border-spacing:0}td,th{padding:0}
/*!************************************************
* Blogger Template Style
* Name: Contempo
**************************************************/
body{
overflow-wrap:break-word;
word-break:break-word;
word-wrap:break-word
}
.hidden{
display:none
}
.invisible{
visibility:hidden
}
.container::after,.float-container::after{
clear:both;
content:"";
display:table
}
.clearboth{
clear:both
}
#comments .comment .comment-actions,.subscribe-popup .FollowByEmail .follow-by-email-submit,.widget.Profile .profile-link,.widget.Profile .profile-link.visit-profile{
background:0 0;
border:0;
box-shadow:none;
color:#2196f3;
cursor:pointer;
font-size:14px;
font-weight:700;
outline:0;
text-decoration:none;
text-transform:uppercase;
width:auto
}
.dim-overlay{
background-color:rgba(0,0,0,.54);
height:100vh;
left:0;
position:fixed;
top:0;
width:100%
}
#sharing-dim-overlay{
background-color:transparent
}
input::-ms-clear{
display:none
}
.blogger-logo,.svg-icon-24.blogger-logo{
fill:#ff9800;
opacity:1
}
.loading-spinner-large{
-webkit-animation:mspin-rotate 1.568s infinite linear;
animation:mspin-rotate 1.568s infinite linear;
height:48px;
overflow:hidden;
position:absolute;
width:48px;
z-index:200
}
.loading-spinner-large>div{
-webkit-animation:mspin-revrot 5332ms infinite steps(4);
animation:mspin-revrot 5332ms infinite steps(4)
}
.loading-spinner-large>div>div{
-webkit-animation:mspin-singlecolor-large-film 1333ms infinite steps(81);
animation:mspin-singlecolor-large-film 1333ms infinite steps(81);
background-size:100%;
height:48px;
width:3888px
}
.mspin-black-large>div>div,.mspin-grey_54-large>div>div{
background-image:url(https://www.blogblog.com/indie/mspin_black_large.svg)
}
.mspin-white-large>div>div{
background-image:url(https://www.blogblog.com/indie/mspin_white_large.svg)
}
.mspin-grey_54-large{
opacity:.54
}
@-webkit-keyframes mspin-singlecolor-large-film{
from{
-webkit-transform:translateX(0);
transform:translateX(0)
}
to{
-webkit-transform:translateX(-3888px);
transform:translateX(-3888px)
}
}
@keyframes mspin-singlecolor-large-film{
from{
-webkit-transform:translateX(0);
transform:translateX(0)
}
to{
-webkit-transform:translateX(-3888px);
transform:translateX(-3888px)
}
}
@-webkit-keyframes mspin-rotate{
from{
-webkit-transform:rotate(0);
transform:rotate(0)
}
to{
-webkit-transform:rotate(360deg);
transform:rotate(360deg)
}
}
@keyframes mspin-rotate{
from{
-webkit-transform:rotate(0);
transform:rotate(0)
}
to{
-webkit-transform:rotate(360deg);
transform:rotate(360deg)
}
}
@-webkit-keyframes mspin-revrot{
from{
-webkit-transform:rotate(0);
transform:rotate(0)
}
to{
-webkit-transform:rotate(-360deg);
transform:rotate(-360deg)
}
}
@keyframes mspin-revrot{
from{
-webkit-transform:rotate(0);
transform:rotate(0)
}
to{
-webkit-transform:rotate(-360deg);
transform:rotate(-360deg)
}
}
.skip-navigation{
background-color:#fff;
box-sizing:border-box;
color:#000;
display:block;
height:0;
left:0;
line-height:50px;
overflow:hidden;
padding-top:0;
position:fixed;
text-align:center;
top:0;
-webkit-transition:box-shadow .3s,height .3s,padding-top .3s;
transition:box-shadow .3s,height .3s,padding-top .3s;
width:100%;
z-index:900
}
.skip-navigation:focus{
box-shadow:0 4px 5px 0 rgba(0,0,0,.14),0 1px 10px 0 rgba(0,0,0,.12),0 2px 4px -1px rgba(0,0,0,.2);
height:50px
}
#main{
outline:0
}
.main-heading{
position:absolute;
clip:rect(1px,1px,1px,1px);
padding:0;
border:0;
height:1px;
width:1px;
overflow:hidden
}
.Attribution{
margin-top:1em;
text-align:center
}
.Attribution .blogger img,.Attribution .blogger svg{
vertical-align:bottom
}
.Attribution .blogger img{
margin-right:.5em
}
.Attribution div{
line-height:24px;
margin-top:.5em
}
.Attribution .copyright,.Attribution .image-attribution{
font-size:.7em;
margin-top:1.5em
}
.BLOG_mobile_video_class{
display:none
}
.bg-photo{
background-attachment:scroll!important
}
body .CSS_LIGHTBOX{
z-index:900
}
.extendable .show-less,.extendable .show-more{
border-color:#2196f3;
color:#2196f3;
margin-top:8px
}
.extendable .show-less.hidden,.extendable .show-more.hidden{
display:none
}
.inline-ad{
display:none;
max-width:100%;
overflow:hidden
}
.adsbygoogle{
display:block
}
#cookieChoiceInfo{
bottom:0;
top:auto
}
iframe.b-hbp-video{
border:0
}
.post-body img{
max-width:100%
}
.post-body iframe{
max-width:100%
}
.post-body a[imageanchor="1"]{
display:inline-block
}
.byline{
margin-right:1em
}
.byline:last-child{
margin-right:0
}
.link-copied-dialog{
max-width:520px;
outline:0
}
.link-copied-dialog .modal-dialog-buttons{
margin-top:8px
}
.link-copied-dialog .goog-buttonset-default{
background:0 0;
border:0
}
.link-copied-dialog .goog-buttonset-default:focus{
outline:0
}
.paging-control-container{
margin-bottom:16px
}
.paging-control-container .paging-control{
display:inline-block
}
.paging-control-container .comment-range-text::after,.paging-control-container .paging-control{
color:#2196f3
}
.paging-control-container .comment-range-text,.paging-control-container .paging-control{
margin-right:8px
}
.paging-control-container .comment-range-text::after,.paging-control-container .paging-control::after{
content:"\b7";
cursor:default;
padding-left:8px;
pointer-events:none
}
.paging-control-container .comment-range-text:last-child::after,.paging-control-container .paging-control:last-child::after{
content:none
}
.byline.reactions iframe{
height:20px
}
.b-notification{
color:#000;
background-color:#fff;
border-bottom:solid 1px #000;
box-sizing:border-box;
padding:16px 32px;
text-align:center
}
.b-notification.visible{
-webkit-transition:margin-top .3s cubic-bezier(.4,0,.2,1);
transition:margin-top .3s cubic-bezier(.4,0,.2,1)
}
.b-notification.invisible{
position:absolute
}
.b-notification-close{
position:absolute;
right:8px;
top:8px
}
.no-posts-message{
line-height:40px;
text-align:center
}
@media screen and (max-width:800px){
body.item-view .post-body a[imageanchor="1"][style*="float: left;"],body.item-view .post-body a[imageanchor="1"][style*="float: right;"]{
float:none!important;
clear:none!important
}
body.item-view .post-body a[imageanchor="1"] img{
display:block;
height:auto;
margin:0 auto
}
body.item-view .post-body>.separator:first-child>a[imageanchor="1"]:first-child{
margin-top:20px
}
.post-body a[imageanchor]{
display:block
}
body.item-view .post-body a[imageanchor="1"]{
margin-left:0!important;
margin-right:0!important
}
body.item-view .post-body a[imageanchor="1"]+a[imageanchor="1"]{
margin-top:16px
}
}
.item-control{
display:none
}
#comments{
border-top:1px dashed rgba(0,0,0,.54);
margin-top:20px;
padding:20px
}
#comments .comment-thread ol{
margin:0;
padding-left:0;
padding-left:0
}
#comments .comment .comment-replybox-single,#comments .comment-thread .comment-replies{
margin-left:60px
}
#comments .comment-thread .thread-count{
display:none
}
#comments .comment{
list-style-type:none;
padding:0 0 30px;
position:relative
}
#comments .comment .comment{
padding-bottom:8px
}
.comment .avatar-image-container{
position:absolute
}
.comment .avatar-image-container img{
border-radius:50%
}
.avatar-image-container svg,.comment .avatar-image-container .avatar-icon{
border-radius:50%;
border:solid 1px #707070;
box-sizing:border-box;
fill:#707070;
height:35px;
margin:0;
padding:7px;
width:35px
}
.comment .comment-block{
margin-top:10px;
margin-left:60px;
padding-bottom:0
}
#comments .comment-author-header-wrapper{
margin-left:40px
}
#comments .comment .thread-expanded .comment-block{
padding-bottom:20px
}
#comments .comment .comment-header .user,#comments .comment .comment-header .user a{
color:#212121;
font-style:normal;
font-weight:700
}
#comments .comment .comment-actions{
bottom:0;
margin-bottom:15px;
position:absolute
}
#comments .comment .comment-actions>*{
margin-right:8px
}
#comments .comment .comment-header .datetime{
bottom:0;
color:rgba(33,33,33,.54);
display:inline-block;
font-size:13px;
font-style:italic;
margin-left:8px
}
#comments .comment .comment-footer .comment-timestamp a,#comments .comment .comment-header .datetime a{
color:rgba(33,33,33,.54)
}
#comments .comment .comment-content,.comment .comment-body{
margin-top:12px;
word-break:break-word
}
.comment-body{
margin-bottom:12px
}
#comments.embed[data-num-comments="0"]{
border:0;
margin-top:0;
padding-top:0
}
#comments.embed[data-num-comments="0"] #comment-post-message,#comments.embed[data-num-comments="0"] div.comment-form>p,#comments.embed[data-num-comments="0"] p.comment-footer{
display:none
}
#comment-editor-src{
display:none
}
.comments .comments-content .loadmore.loaded{
max-height:0;
opacity:0;
overflow:hidden
}
.extendable .remaining-items{
height:0;
overflow:hidden;
-webkit-transition:height .3s cubic-bezier(.4,0,.2,1);
transition:height .3s cubic-bezier(.4,0,.2,1)
}
.extendable .remaining-items.expanded{
height:auto
}
.svg-icon-24,.svg-icon-24-button{
cursor:pointer;
height:24px;
width:24px;
min-width:24px
}
.touch-icon{
margin:-12px;
padding:12px
}
.touch-icon:active,.touch-icon:focus{
background-color:rgba(153,153,153,.4);
border-radius:50%
}
svg:not(:root).touch-icon{
overflow:visible
}
html[dir=rtl] .rtl-reversible-icon{
-webkit-transform:scaleX(-1);
-ms-transform:scaleX(-1);
transform:scaleX(-1)
}
.svg-icon-24-button,.touch-icon-button{
background:0 0;
border:0;
margin:0;
outline:0;
padding:0
}
.touch-icon-button .touch-icon:active,.touch-icon-button .touch-icon:focus{
background-color:transparent
}
.touch-icon-button:active .touch-icon,.touch-icon-button:focus .touch-icon{
background-color:rgba(153,153,153,.4);
border-radius:50%
}
.Profile .default-avatar-wrapper .avatar-icon{
border-radius:50%;
border:solid 1px #707070;
box-sizing:border-box;
fill:#707070;
margin:0
}
.Profile .individual .default-avatar-wrapper .avatar-icon{
padding:25px
}
.Profile .individual .avatar-icon,.Profile .individual .profile-img{
height:120px;
width:120px
}
.Profile .team .default-avatar-wrapper .avatar-icon{
padding:8px
}
.Profile .team .avatar-icon,.Profile .team .default-avatar-wrapper,.Profile .team .profile-img{
height:40px;
width:40px
}
.snippet-container{
margin:0;
position:relative;
overflow:hidden
}
.snippet-fade{
bottom:0;
box-sizing:border-box;
position:absolute;
width:96px
}
.snippet-fade{
right:0
}
.snippet-fade:after{
content:"\2026"
}
.snippet-fade:after{
float:right
}
.post-bottom{
-webkit-box-align:center;
-webkit-align-items:center;
-ms-flex-align:center;
align-items:center;
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex;
-webkit-flex-wrap:wrap;
-ms-flex-wrap:wrap;
flex-wrap:wrap
}
.post-footer{
-webkit-box-flex:1;
-webkit-flex:1 1 auto;
-ms-flex:1 1 auto;
flex:1 1 auto;
-webkit-flex-wrap:wrap;
-ms-flex-wrap:wrap;
flex-wrap:wrap;
-webkit-box-ordinal-group:2;
-webkit-order:1;
-ms-flex-order:1;
order:1
}
.post-footer>*{
-webkit-box-flex:0;
-webkit-flex:0 1 auto;
-ms-flex:0 1 auto;
flex:0 1 auto
}
.post-footer .byline:last-child{
margin-right:1em
}
.jump-link{
-webkit-box-flex:0;
-webkit-flex:0 0 auto;
-ms-flex:0 0 auto;
flex:0 0 auto;
-webkit-box-ordinal-group:3;
-webkit-order:2;
-ms-flex-order:2;
order:2
}
.centered-top-container.sticky{
left:0;
position:fixed;
right:0;
top:0;
width:auto;
z-index:50;
-webkit-transition-property:opacity,-webkit-transform;
transition-property:opacity,-webkit-transform;
transition-property:transform,opacity;
transition-property:transform,opacity,-webkit-transform;
-webkit-transition-duration:.2s;
transition-duration:.2s;
-webkit-transition-timing-function:cubic-bezier(.4,0,.2,1);
transition-timing-function:cubic-bezier(.4,0,.2,1)
}
.centered-top-placeholder{
display:none
}
.collapsed-header .centered-top-placeholder{
display:block
}
.centered-top-container .Header .replaced h1,.centered-top-placeholder .Header .replaced h1{
display:none
}
.centered-top-container.sticky .Header .replaced h1{
display:block
}
.centered-top-container.sticky .Header .header-widget{
background:0 0
}
.centered-top-container.sticky .Header .header-image-wrapper{
display:none
}
.centered-top-container img,.centered-top-placeholder img{
max-width:100%
}
.collapsible{
-webkit-transition:height .3s cubic-bezier(.4,0,.2,1);
transition:height .3s cubic-bezier(.4,0,.2,1)
}
.collapsible,.collapsible>summary{
display:block;
overflow:hidden
}
.collapsible>:not(summary){
display:none
}
.collapsible[open]>:not(summary){
display:block
}
.collapsible:focus,.collapsible>summary:focus{
outline:0
}
.collapsible>summary{
cursor:pointer;
display:block;
padding:0
}
.collapsible:focus>summary,.collapsible>summary:focus{
background-color:transparent
}
.collapsible>summary::-webkit-details-marker{
display:none
}
.collapsible-title{
-webkit-box-align:center;
-webkit-align-items:center;
-ms-flex-align:center;
align-items:center;
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex
}
.collapsible-title .title{
-webkit-box-flex:1;
-webkit-flex:1 1 auto;
-ms-flex:1 1 auto;
flex:1 1 auto;
-webkit-box-ordinal-group:1;
-webkit-order:0;
-ms-flex-order:0;
order:0;
overflow:hidden;
text-overflow:ellipsis;
white-space:nowrap
}
.collapsible-title .chevron-down,.collapsible[open] .collapsible-title .chevron-up{
display:block
}
.collapsible-title .chevron-up,.collapsible[open] .collapsible-title .chevron-down{
display:none
}
.flat-button{
cursor:pointer;
display:inline-block;
font-weight:700;
text-transform:uppercase;
border-radius:2px;
padding:8px;
margin:-8px
}
.flat-icon-button{
background:0 0;
border:0;
margin:0;
outline:0;
padding:0;
margin:-12px;
padding:12px;
cursor:pointer;
box-sizing:content-box;
display:inline-block;
line-height:0
}
.flat-icon-button,.flat-icon-button .splash-wrapper{
border-radius:50%
}
.flat-icon-button .splash.animate{
-webkit-animation-duration:.3s;
animation-duration:.3s
}
.overflowable-container{
max-height:46px;
overflow:hidden;
position:relative
}
.overflow-button{
cursor:pointer
}
#overflowable-dim-overlay{
background:0 0
}
.overflow-popup{
box-shadow:0 2px 2px 0 rgba(0,0,0,.14),0 3px 1px -2px rgba(0,0,0,.2),0 1px 5px 0 rgba(0,0,0,.12);
background-color:#ffffff;
left:0;
max-width:calc(100% - 32px);
position:absolute;
top:0;
visibility:hidden;
z-index:101
}
.overflow-popup ul{
list-style:none
}
.overflow-popup .tabs li,.overflow-popup li{
display:block;
height:auto
}
.overflow-popup .tabs li{
padding-left:0;
padding-right:0
}
.overflow-button.hidden,.overflow-popup .tabs li.hidden,.overflow-popup li.hidden{
display:none
}
.pill-button{
background:0 0;
border:1px solid;
border-radius:12px;
cursor:pointer;
display:inline-block;
padding:4px 16px;
text-transform:uppercase
}
.ripple{
position:relative
}
.ripple>*{
z-index:1
}
.splash-wrapper{
bottom:0;
left:0;
overflow:hidden;
pointer-events:none;
position:absolute;
right:0;
top:0;
z-index:0
}
.splash{
background:#ccc;
border-radius:100%;
display:block;
opacity:.6;
position:absolute;
-webkit-transform:scale(0);
-ms-transform:scale(0);
transform:scale(0)
}
.splash.animate{
-webkit-animation:ripple-effect .4s linear;
animation:ripple-effect .4s linear
}
@-webkit-keyframes ripple-effect{
100%{
opacity:0;
-webkit-transform:scale(2.5);
transform:scale(2.5)
}
}
@keyframes ripple-effect{
100%{
opacity:0;
-webkit-transform:scale(2.5);
transform:scale(2.5)
}
}
.search{
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex;
line-height:24px;
width:24px
}
.search.focused{
width:100%
}
.search.focused .section{
width:100%
}
.search form{
z-index:101
}
.search h3{
display:none
}
.search form{
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex;
-webkit-box-flex:1;
-webkit-flex:1 0 0;
-ms-flex:1 0 0px;
flex:1 0 0;
border-bottom:solid 1px transparent;
padding-bottom:8px
}
.search form>*{
display:none
}
.search.focused form>*{
display:block
}
.search .search-input label{
display:none
}
.centered-top-placeholder.cloned .search form{
z-index:30
}
.search.focused form{
border-color:#ffffff;
position:relative;
width:auto
}
.collapsed-header .centered-top-container .search.focused form{
border-bottom-color:transparent
}
.search-expand{
-webkit-box-flex:0;
-webkit-flex:0 0 auto;
-ms-flex:0 0 auto;
flex:0 0 auto
}
.search-expand-text{
display:none
}
.search-close{
display:inline;
vertical-align:middle
}
.search-input{
-webkit-box-flex:1;
-webkit-flex:1 0 1px;
-ms-flex:1 0 1px;
flex:1 0 1px
}
.search-input input{
background:0 0;
border:0;
box-sizing:border-box;
color:#ffffff;
display:inline-block;
outline:0;
width:calc(100% - 48px)
}
.search-input input.no-cursor{
color:transparent;
text-shadow:0 0 0 #ffffff
}
.collapsed-header .centered-top-container .search-action,.collapsed-header .centered-top-container .search-input input{
color:#212121
}
.collapsed-header .centered-top-container .search-input input.no-cursor{
color:transparent;
text-shadow:0 0 0 #212121
}
.collapsed-header .centered-top-container .search-input input.no-cursor:focus,.search-input input.no-cursor:focus{
outline:0
}
.search-focused>*{
visibility:hidden
}
.search-focused .search,.search-focused .search-icon{
visibility:visible
}
.search.focused .search-action{
display:block
}
.search.focused .search-action:disabled{
opacity:.3
}
.widget.Sharing .sharing-button{
display:none
}
.widget.Sharing .sharing-buttons li{
padding:0
}
.widget.Sharing .sharing-buttons li span{
display:none
}
.post-share-buttons{
position:relative
}
.centered-bottom .share-buttons .svg-icon-24,.share-buttons .svg-icon-24{
fill:#212121
}
.sharing-open.touch-icon-button:active .touch-icon,.sharing-open.touch-icon-button:focus .touch-icon{
background-color:transparent
}
.share-buttons{
background-color:#ffffff;
border-radius:2px;
box-shadow:0 2px 2px 0 rgba(0,0,0,.14),0 3px 1px -2px rgba(0,0,0,.2),0 1px 5px 0 rgba(0,0,0,.12);
color:#212121;
list-style:none;
margin:0;
padding:8px 0;
position:absolute;
top:-11px;
min-width:200px;
z-index:101
}
.share-buttons.hidden{
display:none
}
.sharing-button{
background:0 0;
border:0;
margin:0;
outline:0;
padding:0;
cursor:pointer
}
.share-buttons li{
margin:0;
height:48px
}
.share-buttons li:last-child{
margin-bottom:0
}
.share-buttons li .sharing-platform-button{
box-sizing:border-box;
cursor:pointer;
display:block;
height:100%;
margin-bottom:0;
padding:0 16px;
position:relative;
width:100%
}
.share-buttons li .sharing-platform-button:focus,.share-buttons li .sharing-platform-button:hover{
background-color:rgba(128,128,128,.1);
outline:0
}
.share-buttons li svg[class*=" sharing-"],.share-buttons li svg[class^=sharing-]{
position:absolute;
top:10px
}
.share-buttons li span.sharing-platform-button{
position:relative;
top:0
}
.share-buttons li .platform-sharing-text{
display:block;
font-size:16px;
line-height:48px;
white-space:nowrap
}
.share-buttons li .platform-sharing-text{
margin-left:56px
}
.sidebar-container{
background-color:#ffffff;
max-width:284px;
overflow-y:auto;
-webkit-transition-property:-webkit-transform;
transition-property:-webkit-transform;
transition-property:transform;
transition-property:transform,-webkit-transform;
-webkit-transition-duration:.3s;
transition-duration:.3s;
-webkit-transition-timing-function:cubic-bezier(0,0,.2,1);
transition-timing-function:cubic-bezier(0,0,.2,1);
width:284px;
z-index:101;
-webkit-overflow-scrolling:touch
}
.sidebar-container .navigation{
line-height:0;
padding:16px
}
.sidebar-container .sidebar-back{
cursor:pointer
}
.sidebar-container .widget{
background:0 0;
margin:0 16px;
padding:16px 0
}
.sidebar-container .widget .title{
color:#212121;
margin:0
}
.sidebar-container .widget ul{
list-style:none;
margin:0;
padding:0
}
.sidebar-container .widget ul ul{
margin-left:1em
}
.sidebar-container .widget li{
font-size:16px;
line-height:normal
}
.sidebar-container .widget+.widget{
border-top:1px dashed #cccccc
}
.BlogArchive li{
margin:16px 0
}
.BlogArchive li:last-child{
margin-bottom:0
}
.Label li a{
display:inline-block
}
.BlogArchive .post-count,.Label .label-count{
float:right;
margin-left:.25em
}
.BlogArchive .post-count::before,.Label .label-count::before{
content:"("
}
.BlogArchive .post-count::after,.Label .label-count::after{
content:")"
}
.widget.Translate .skiptranslate>div{
display:block!important
}
.widget.Profile .profile-link{
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex
}
.widget.Profile .team-member .default-avatar-wrapper,.widget.Profile .team-member .profile-img{
-webkit-box-flex:0;
-webkit-flex:0 0 auto;
-ms-flex:0 0 auto;
flex:0 0 auto;
margin-right:1em
}
.widget.Profile .individual .profile-link{
-webkit-box-orient:vertical;
-webkit-box-direction:normal;
-webkit-flex-direction:column;
-ms-flex-direction:column;
flex-direction:column
}
.widget.Profile .team .profile-link .profile-name{
-webkit-align-self:center;
-ms-flex-item-align:center;
align-self:center;
display:block;
-webkit-box-flex:1;
-webkit-flex:1 1 auto;
-ms-flex:1 1 auto;
flex:1 1 auto
}
.dim-overlay{
background-color:rgba(0,0,0,.54);
z-index:100
}
body.sidebar-visible{
overflow-y:hidden
}
@media screen and (max-width:1439px){
.sidebar-container{
bottom:0;
position:fixed;
top:0;
left:0;
right:auto
}
.sidebar-container.sidebar-invisible{
-webkit-transition-timing-function:cubic-bezier(.4,0,.6,1);
transition-timing-function:cubic-bezier(.4,0,.6,1);
-webkit-transform:translateX(-284px);
-ms-transform:translateX(-284px);
transform:translateX(-284px)
}
}
@media screen and (min-width:1440px){
.sidebar-container{
position:absolute;
top:0;
left:0;
right:auto
}
.sidebar-container .navigation{
display:none
}
}
.dialog{
box-shadow:0 2px 2px 0 rgba(0,0,0,.14),0 3px 1px -2px rgba(0,0,0,.2),0 1px 5px 0 rgba(0,0,0,.12);
background:#ffffff;
box-sizing:border-box;
color:#757575;
padding:30px;
position:fixed;
text-align:center;
width:calc(100% - 24px);
z-index:101
}
.dialog input[type=email],.dialog input[type=text]{
background-color:transparent;
border:0;
border-bottom:solid 1px rgba(117,117,117,.12);
color:#757575;
display:block;
font-family:Roboto, sans-serif;
font-size:16px;
line-height:24px;
margin:auto;
padding-bottom:7px;
outline:0;
text-align:center;
width:100%
}
.dialog input[type=email]::-webkit-input-placeholder,.dialog input[type=text]::-webkit-input-placeholder{
color:#757575
}
.dialog input[type=email]::-moz-placeholder,.dialog input[type=text]::-moz-placeholder{
color:#757575
}
.dialog input[type=email]:-ms-input-placeholder,.dialog input[type=text]:-ms-input-placeholder{
color:#757575
}
.dialog input[type=email]::-ms-input-placeholder,.dialog input[type=text]::-ms-input-placeholder{
color:#757575
}
.dialog input[type=email]::placeholder,.dialog input[type=text]::placeholder{
color:#757575
}
.dialog input[type=email]:focus,.dialog input[type=text]:focus{
border-bottom:solid 2px #2196f3;
padding-bottom:6px
}
.dialog input.no-cursor{
color:transparent;
text-shadow:0 0 0 #757575
}
.dialog input.no-cursor:focus{
outline:0
}
.dialog input.no-cursor:focus{
outline:0
}
.dialog input[type=submit]{
font-family:Roboto, sans-serif
}
.dialog .goog-buttonset-default{
color:#2196f3
}
.subscribe-popup{
max-width:364px
}
.subscribe-popup h3{
color:#212121;
font-size:1.8em;
margin-top:0
}
.subscribe-popup .FollowByEmail h3{
display:none
}
.subscribe-popup .FollowByEmail .follow-by-email-submit{
color:#2196f3;
display:inline-block;
margin:0 auto;
margin-top:24px;
width:auto;
white-space:normal
}
.subscribe-popup .FollowByEmail .follow-by-email-submit:disabled{
cursor:default;
opacity:.3
}
@media (max-width:800px){
.blog-name div.widget.Subscribe{
margin-bottom:16px
}
body.item-view .blog-name div.widget.Subscribe{
margin:8px auto 16px auto;
width:100%
}
}
.tabs{
list-style:none
}
.tabs li{
display:inline-block
}
.tabs li a{
cursor:pointer;
display:inline-block;
font-weight:700;
text-transform:uppercase;
padding:12px 8px
}
.tabs .selected{
border-bottom:4px solid #ffffff
}
.tabs .selected a{
color:#ffffff
}
body#layout .bg-photo,body#layout .bg-photo-overlay{
display:none
}
body#layout .page_body{
padding:0;
position:relative;
top:0
}
body#layout .page{
display:inline-block;
left:inherit;
position:relative;
vertical-align:top;
width:540px
}
body#layout .centered{
max-width:954px
}
body#layout .navigation{
display:none
}
body#layout .sidebar-container{
display:inline-block;
width:40%
}
body#layout .hamburger-menu,body#layout .search{
display:none
}
.centered-top-container .svg-icon-24,body.collapsed-header .centered-top-placeholder .svg-icon-24{
fill:#ffffff
}
.sidebar-container .svg-icon-24{
fill:#707070
}
.centered-bottom .svg-icon-24,body.collapsed-header .centered-top-container .svg-icon-24{
fill:#707070
}
.centered-bottom .share-buttons .svg-icon-24,.share-buttons .svg-icon-24{
fill:#212121
}
body{
background-color:#eeeeee;
color:#757575;
font:15px Roboto, sans-serif;
margin:0;
min-height:100vh
}
img{
max-width:100%
}
h3{
color:#757575;
font-size:16px
}
a{
text-decoration:none;
color:#2196f3
}
a:visited{
color:#2196f3
}
a:hover{
color:#2196f3
}
blockquote{
color:#444444;
font:italic 300 15px Roboto, sans-serif;
font-size:x-large;
text-align:center
}
.pill-button{
font-size:12px
}
.bg-photo-container{
height:480px;
overflow:hidden;
position:absolute;
width:100%;
z-index:1
}
.bg-photo{
background:#eeeeee url(https://themes.googleusercontent.com/image?id=L1lcAxxz0CLgsDzixEprHJ2F38TyEjCyE3RSAjynQDks0lT1BDc1OxXKaTEdLc89HPvdB11X9FDw) no-repeat scroll top center /* Credit: Michael Elkan (http://www.offset.com/photos/394244) */;;
background-attachment:scroll;
background-size:cover;
-webkit-filter:blur(0px);
filter:blur(0px);
height:calc(100% + 2 * 0px);
left:0px;
position:absolute;
top:0px;
width:calc(100% + 2 * 0px)
}
.bg-photo-overlay{
background:rgba(0,0,0,.26);
background-size:cover;
height:480px;
position:absolute;
width:100%;
z-index:2
}
.hamburger-menu{
float:left;
margin-top:0
}
.sticky .hamburger-menu{
float:none;
position:absolute
}
.search{
border-bottom:solid 1px rgba(255, 255, 255, 0);
float:right;
position:relative;
-webkit-transition-property:width;
transition-property:width;
-webkit-transition-duration:.5s;
transition-duration:.5s;
-webkit-transition-timing-function:cubic-bezier(.4,0,.2,1);
transition-timing-function:cubic-bezier(.4,0,.2,1);
z-index:101
}
.search .dim-overlay{
background-color:transparent
}
.search form{
height:36px;
-webkit-transition-property:border-color;
transition-property:border-color;
-webkit-transition-delay:.5s;
transition-delay:.5s;
-webkit-transition-duration:.2s;
transition-duration:.2s;
-webkit-transition-timing-function:cubic-bezier(.4,0,.2,1);
transition-timing-function:cubic-bezier(.4,0,.2,1)
}
.search.focused{
width:calc(100% - 48px)
}
.search.focused form{
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex;
-webkit-box-flex:1;
-webkit-flex:1 0 1px;
-ms-flex:1 0 1px;
flex:1 0 1px;
border-color:#ffffff;
margin-left:-24px;
padding-left:36px;
position:relative;
width:auto
}
.item-view .search,.sticky .search{
right:0;
float:none;
margin-left:0;
position:absolute
}
.item-view .search.focused,.sticky .search.focused{
width:calc(100% - 50px)
}
.item-view .search.focused form,.sticky .search.focused form{
border-bottom-color:#757575
}
.centered-top-placeholder.cloned .search form{
z-index:30
}
.search_button{
-webkit-box-flex:0;
-webkit-flex:0 0 24px;
-ms-flex:0 0 24px;
flex:0 0 24px;
-webkit-box-orient:vertical;
-webkit-box-direction:normal;
-webkit-flex-direction:column;
-ms-flex-direction:column;
flex-direction:column
}
.search_button svg{
margin-top:0
}
.search-input{
height:48px
}
.search-input input{
display:block;
color:#ffffff;
font:16px Roboto, sans-serif;
height:48px;
line-height:48px;
padding:0;
width:100%
}
.search-input input::-webkit-input-placeholder{
color:#ffffff;
opacity:.3
}
.search-input input::-moz-placeholder{
color:#ffffff;
opacity:.3
}
.search-input input:-ms-input-placeholder{
color:#ffffff;
opacity:.3
}
.search-input input::-ms-input-placeholder{
color:#ffffff;
opacity:.3
}
.search-input input::placeholder{
color:#ffffff;
opacity:.3
}
.search-action{
background:0 0;
border:0;
color:#ffffff;
cursor:pointer;
display:none;
height:48px;
margin-top:0
}
.sticky .search-action{
color:#757575
}
.search.focused .search-action{
display:block
}
.search.focused .search-action:disabled{
opacity:.3
}
.page_body{
position:relative;
z-index:20
}
.page_body .widget{
margin-bottom:16px
}
.page_body .centered{
box-sizing:border-box;
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex;
-webkit-box-orient:vertical;
-webkit-box-direction:normal;
-webkit-flex-direction:column;
-ms-flex-direction:column;
flex-direction:column;
margin:0 auto;
max-width:922px;
min-height:100vh;
padding:24px 0
}
.page_body .centered>*{
-webkit-box-flex:0;
-webkit-flex:0 0 auto;
-ms-flex:0 0 auto;
flex:0 0 auto
}
.page_body .centered>#footer{
margin-top:auto
}
.blog-name{
margin:24px 0 16px 0
}
.item-view .blog-name,.sticky .blog-name{
box-sizing:border-box;
margin-left:36px;
min-height:48px;
opacity:1;
padding-top:12px
}
.blog-name .subscribe-section-container{
margin-bottom:32px;
text-align:center;
-webkit-transition-property:opacity;
transition-property:opacity;
-webkit-transition-duration:.5s;
transition-duration:.5s
}
.item-view .blog-name .subscribe-section-container,.sticky .blog-name .subscribe-section-container{
margin:0 0 8px 0
}
.blog-name .PageList{
margin-top:16px;
padding-top:8px;
text-align:center
}
.blog-name .PageList .overflowable-contents{
width:100%
}
.blog-name .PageList h3.title{
color:#ffffff;
margin:8px auto;
text-align:center;
width:100%
}
.centered-top-container .blog-name{
-webkit-transition-property:opacity;
transition-property:opacity;
-webkit-transition-duration:.5s;
transition-duration:.5s
}
.item-view .return_link{
margin-bottom:12px;
margin-top:12px;
position:absolute
}
.item-view .blog-name{
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex;
-webkit-flex-wrap:wrap;
-ms-flex-wrap:wrap;
flex-wrap:wrap;
margin:0 48px 27px 48px
}
.item-view .subscribe-section-container{
-webkit-box-flex:0;
-webkit-flex:0 0 auto;
-ms-flex:0 0 auto;
flex:0 0 auto
}
.item-view #header,.item-view .Header{
margin-bottom:5px;
margin-right:15px
}
.item-view .sticky .Header{
margin-bottom:0
}
.item-view .Header p{
margin:10px 0 0 0;
text-align:left
}
.item-view .post-share-buttons-bottom{
margin-right:16px
}
.sticky{
background:#ffffff;
box-shadow:0 0 20px 0 rgba(0,0,0,.7);
box-sizing:border-box;
margin-left:0
}
.sticky #header{
margin-bottom:8px;
margin-right:8px
}
.sticky .centered-top{
margin:4px auto;
max-width:890px;
min-height:48px
}
.sticky .blog-name{
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex;
margin:0 48px
}
.sticky .blog-name #header{
-webkit-box-flex:0;
-webkit-flex:0 1 auto;
-ms-flex:0 1 auto;
flex:0 1 auto;
-webkit-box-ordinal-group:2;
-webkit-order:1;
-ms-flex-order:1;
order:1;
overflow:hidden
}
.sticky .blog-name .subscribe-section-container{
-webkit-box-flex:0;
-webkit-flex:0 0 auto;
-ms-flex:0 0 auto;
flex:0 0 auto;
-webkit-box-ordinal-group:3;
-webkit-order:2;
-ms-flex-order:2;
order:2
}
.sticky .Header h1{
overflow:hidden;
text-overflow:ellipsis;
white-space:nowrap;
margin-right:-10px;
margin-bottom:-10px;
padding-right:10px;
padding-bottom:10px
}
.sticky .Header p{
display:none
}
.sticky .PageList{
display:none
}
.search-focused>*{
visibility:visible
}
.search-focused .hamburger-menu{
visibility:visible
}
.item-view .search-focused .blog-name,.sticky .search-focused .blog-name{
opacity:0
}
.centered-bottom,.centered-top-container,.centered-top-placeholder{
padding:0 16px
}
.centered-top{
position:relative
}
.item-view .centered-top.search-focused .subscribe-section-container,.sticky .centered-top.search-focused .subscribe-section-container{
opacity:0
}
.page_body.has-vertical-ads .centered .centered-bottom{
display:inline-block;
width:calc(100% - 176px)
}
.Header h1{
color:#ffffff;
font:bold 45px Roboto, sans-serif;
line-height:normal;
margin:0 0 13px 0;
text-align:center;
width:100%
}
.Header h1 a,.Header h1 a:hover,.Header h1 a:visited{
color:#ffffff
}
.item-view .Header h1,.sticky .Header h1{
font-size:24px;
line-height:24px;
margin:0;
text-align:left
}
.sticky .Header h1{
color:#757575
}
.sticky .Header h1 a,.sticky .Header h1 a:hover,.sticky .Header h1 a:visited{
color:#757575
}
.Header p{
color:#ffffff;
margin:0 0 13px 0;
opacity:.8;
text-align:center
}
.widget .title{
line-height:28px
}
.BlogArchive li{
font-size:16px
}
.BlogArchive .post-count{
color:#757575
}
#page_body .FeaturedPost,.Blog .blog-posts .post-outer-container{
background:#ffffff;
min-height:40px;
padding:30px 40px;
width:auto
}
.Blog .blog-posts .post-outer-container:last-child{
margin-bottom:0
}
.Blog .blog-posts .post-outer-container .post-outer{
border:0;
position:relative;
padding-bottom:.25em
}
.post-outer-container{
margin-bottom:16px
}
.post:first-child{
margin-top:0
}
.post .thumb{
float:left;
height:20%;
width:20%
}
.post-share-buttons-bottom,.post-share-buttons-top{
float:right
}
.post-share-buttons-bottom{
margin-right:24px
}
.post-footer,.post-header{
clear:left;
color:rgba(0, 0, 0, 0.54);
margin:0;
width:inherit
}
.blog-pager{
text-align:center
}
.blog-pager a{
color:#2196f3
}
.blog-pager a:visited{
color:#2196f3
}
.blog-pager a:hover{
color:#2196f3
}
.post-title{
font:bold 22px Roboto, sans-serif;
float:left;
margin:0 0 8px 0;
max-width:calc(100% - 48px)
}
.post-title a{
font:bold 30px Roboto, sans-serif
}
.post-title,.post-title a,.post-title a:hover,.post-title a:visited{
color:#212121
}
.post-body{
color:#757575;
font:15px Roboto, sans-serif;
line-height:1.6em;
margin:1.5em 0 2em 0;
display:block
}
.post-body img{
height:inherit
}
.post-body .snippet-thumbnail{
float:left;
margin:0;
margin-right:2em;
max-height:128px;
max-width:128px
}
.post-body .snippet-thumbnail img{
max-width:100%
}
.main .FeaturedPost .widget-content{
border:0;
position:relative;
padding-bottom:.25em
}
.FeaturedPost img{
margin-top:2em
}
.FeaturedPost .snippet-container{
margin:2em 0
}
.FeaturedPost .snippet-container p{
margin:0
}
.FeaturedPost .snippet-thumbnail{
float:none;
height:auto;
margin-bottom:2em;
margin-right:0;
overflow:hidden;
max-height:calc(600px + 2em);
max-width:100%;
text-align:center;
width:100%
}
.FeaturedPost .snippet-thumbnail img{
max-width:100%;
width:100%
}
.byline{
color:rgba(0, 0, 0, 0.54);
display:inline-block;
line-height:24px;
margin-top:8px;
vertical-align:top
}
.byline.post-author:first-child{
margin-right:0
}
.byline.reactions .reactions-label{
line-height:22px;
vertical-align:top
}
.byline.post-share-buttons{
position:relative;
display:inline-block;
margin-top:0;
width:100%
}
.byline.post-share-buttons .sharing{
float:right
}
.flat-button.ripple:hover{
background-color:rgba(33,150,243,.12)
}
.flat-button.ripple .splash{
background-color:rgba(33,150,243,.4)
}
a.timestamp-link,a:active.timestamp-link,a:visited.timestamp-link{
color:inherit;
font:inherit;
text-decoration:inherit
}
.post-share-buttons{
margin-left:0
}
.clear-sharing{
min-height:24px
}
.comment-link{
color:#2196f3;
position:relative
}
.comment-link .num_comments{
margin-left:8px;
vertical-align:top
}
#comment-holder .continue{
display:none
}
#comment-editor{
margin-bottom:20px;
margin-top:20px
}
#comments .comment-form h4,#comments h3.title{
position:absolute;
clip:rect(1px,1px,1px,1px);
padding:0;
border:0;
height:1px;
width:1px;
overflow:hidden
}
.post-filter-message{
background-color:rgba(0,0,0,.7);
color:#fff;
display:table;
margin-bottom:16px;
width:100%
}
.post-filter-message div{
display:table-cell;
padding:15px 28px
}
.post-filter-message div:last-child{
padding-left:0;
text-align:right
}
.post-filter-message a{
white-space:nowrap
}
.post-filter-message .search-label,.post-filter-message .search-query{
font-weight:700;
color:#2196f3
}
#blog-pager{
margin:2em 0
}
#blog-pager a{
color:#2196f3;
font-size:14px
}
.subscribe-button{
border-color:#ffffff;
color:#ffffff
}
.sticky .subscribe-button{
border-color:#757575;
color:#757575
}
.tabs{
margin:0 auto;
padding:0
}
.tabs li{
margin:0 8px;
vertical-align:top
}
.tabs .overflow-button a,.tabs li a{
color:#cccccc;
font:700 normal 15px Roboto, sans-serif;
line-height:18px
}
.tabs .overflow-button a{
padding:12px 8px
}
.overflow-popup .tabs li{
text-align:left
}
.overflow-popup li a{
color:#757575;
display:block;
padding:8px 20px
}
.overflow-popup li.selected a{
color:#212121
}
a.report_abuse{
font-weight:400
}
.Label li,.Label span.label-size,.byline.post-labels a{
background-color:#f7f7f7;
border:1px solid #f7f7f7;
border-radius:15px;
display:inline-block;
margin:4px 4px 4px 0;
padding:3px 8px
}
.Label a,.byline.post-labels a{
color:rgba(0,0,0,0.54)
}
.Label ul{
list-style:none;
padding:0
}
.PopularPosts{
background-color:#eeeeee;
padding:30px 40px
}
.PopularPosts .item-content{
color:#757575;
margin-top:24px
}
.PopularPosts a,.PopularPosts a:hover,.PopularPosts a:visited{
color:#2196f3
}
.PopularPosts .post-title,.PopularPosts .post-title a,.PopularPosts .post-title a:hover,.PopularPosts .post-title a:visited{
color:#212121;
font-size:18px;
font-weight:700;
line-height:24px
}
.PopularPosts,.PopularPosts h3.title a{
color:#757575;
font:15px Roboto, sans-serif
}
.main .PopularPosts{
padding:16px 40px
}
.PopularPosts h3.title{
font-size:14px;
margin:0
}
.PopularPosts h3.post-title{
margin-bottom:0
}
.PopularPosts .byline{
color:rgba(0, 0, 0, 0.54)
}
.PopularPosts .jump-link{
float:right;
margin-top:16px
}
.PopularPosts .post-header .byline{
font-size:.9em;
font-style:italic;
margin-top:6px
}
.PopularPosts ul{
list-style:none;
padding:0;
margin:0
}
.PopularPosts .post{
padding:20px 0
}
.PopularPosts .post+.post{
border-top:1px dashed #cccccc
}
.PopularPosts .item-thumbnail{
float:left;
margin-right:32px
}
.PopularPosts .item-thumbnail img{
height:88px;
padding:0;
width:88px
}
.inline-ad{
margin-bottom:16px
}
.desktop-ad .inline-ad{
display:block
}
.adsbygoogle{
overflow:hidden
}
.vertical-ad-container{
float:right;
margin-right:16px;
width:128px
}
.vertical-ad-container .AdSense+.AdSense{
margin-top:16px
}
.inline-ad-placeholder,.vertical-ad-placeholder{
background:#ffffff;
border:1px solid #000;
opacity:.9;
vertical-align:middle;
text-align:center
}
.inline-ad-placeholder span,.vertical-ad-placeholder span{
margin-top:290px;
display:block;
text-transform:uppercase;
font-weight:700;
color:#212121
}
.vertical-ad-placeholder{
height:600px
}
.vertical-ad-placeholder span{
margin-top:290px;
padding:0 40px
}
.inline-ad-placeholder{
height:90px
}
.inline-ad-placeholder span{
margin-top:36px
}
.Attribution{
color:#757575
}
.Attribution a,.Attribution a:hover,.Attribution a:visited{
color:#2196f3
}
.Attribution svg{
fill:#707070
}
.sidebar-container{
box-shadow:1px 1px 3px rgba(0,0,0,.1)
}
.sidebar-container,.sidebar-container .sidebar_bottom{
background-color:#ffffff
}
.sidebar-container .navigation,.sidebar-container .sidebar_top_wrapper{
background-color:#ffffff
}
.sidebar-container .sidebar_top{
overflow:auto
}
.sidebar-container .sidebar_bottom{
width:100%;
padding-top:16px
}
.sidebar-container .widget:first-child{
padding-top:0
}
.sidebar_top .widget.Profile{
padding-bottom:16px
}
.widget.Profile{
margin:0;
width:100%
}
.widget.Profile h2{
display:none
}
.widget.Profile h3.title{
color:rgba(0,0,0,0.52);
margin:16px 32px
}
.widget.Profile .individual{
text-align:center
}
.widget.Profile .individual .profile-link{
padding:1em
}
.widget.Profile .individual .default-avatar-wrapper .avatar-icon{
margin:auto
}
.widget.Profile .team{
margin-bottom:32px;
margin-left:32px;
margin-right:32px
}
.widget.Profile ul{
list-style:none;
padding:0
}
.widget.Profile li{
margin:10px 0
}
.widget.Profile .profile-img{
border-radius:50%;
float:none
}
.widget.Profile .profile-link{
color:#212121;
font-size:.9em;
margin-bottom:1em;
opacity:.87;
overflow:hidden
}
.widget.Profile .profile-link.visit-profile{
border-style:solid;
border-width:1px;
border-radius:12px;
cursor:pointer;
font-size:12px;
font-weight:400;
padding:5px 20px;
display:inline-block;
line-height:normal
}
.widget.Profile dd{
color:rgba(0, 0, 0, 0.54);
margin:0 16px
}
.widget.Profile location{
margin-bottom:1em
}
.widget.Profile .profile-textblock{
font-size:14px;
line-height:24px;
position:relative
}
body.sidebar-visible .page_body{
overflow-y:scroll
}
body.sidebar-visible .bg-photo-container{
overflow-y:scroll
}
@media screen and (min-width:1440px){
.sidebar-container{
margin-top:480px;
min-height:calc(100% - 480px);
overflow:visible;
z-index:32
}
.sidebar-container .sidebar_top_wrapper{
background-color:#f7f7f7;
height:480px;
margin-top:-480px
}
.sidebar-container .sidebar_top{
display:-webkit-box;
display:-webkit-flex;
display:-ms-flexbox;
display:flex;
height:480px;
-webkit-box-orient:horizontal;
-webkit-box-direction:normal;
-webkit-flex-direction:row;
-ms-flex-direction:row;
flex-direction:row;
max-height:480px
}
.sidebar-container .sidebar_bottom{
max-width:284px;
width:284px
}
body.collapsed-header .sidebar-container{
z-index:15
}
.sidebar-container .sidebar_top:empty{
display:none
}
.sidebar-container .sidebar_top>:only-child{
-webkit-box-flex:0;
-webkit-flex:0 0 auto;
-ms-flex:0 0 auto;
flex:0 0 auto;
-webkit-align-self:center;
-ms-flex-item-align:center;
align-self:center;
width:100%
}
.sidebar_top_wrapper.no-items{
display:none
}
}
.post-snippet.snippet-container{
max-height:120px
}
.post-snippet .snippet-item{
line-height:24px
}
.post-snippet .snippet-fade{
background:-webkit-linear-gradient(left,#ffffff 0,#ffffff 20%,rgba(255, 255, 255, 0) 100%);
background:linear-gradient(to left,#ffffff 0,#ffffff 20%,rgba(255, 255, 255, 0) 100%);
color:#757575;
height:24px
}
.popular-posts-snippet.snippet-container{
max-height:72px
}
.popular-posts-snippet .snippet-item{
line-height:24px
}
.PopularPosts .popular-posts-snippet .snippet-fade{
color:#757575;
height:24px
}
.main .popular-posts-snippet .snippet-fade{
background:-webkit-linear-gradient(left,#eeeeee 0,#eeeeee 20%,rgba(238, 238, 238, 0) 100%);
background:linear-gradient(to left,#eeeeee 0,#eeeeee 20%,rgba(238, 238, 238, 0) 100%)
}
.sidebar_bottom .popular-posts-snippet .snippet-fade{
background:-webkit-linear-gradient(left,#ffffff 0,#ffffff 20%,rgba(255, 255, 255, 0) 100%);
background:linear-gradient(to left,#ffffff 0,#ffffff 20%,rgba(255, 255, 255, 0) 100%)
}
.profile-snippet.snippet-container{
max-height:192px
}
.has-location .profile-snippet.snippet-container{
max-height:144px
}
.profile-snippet .snippet-item{
line-height:24px
}
.profile-snippet .snippet-fade{
background:-webkit-linear-gradient(left,#ffffff 0,#ffffff 20%,rgba(255, 255, 255, 0) 100%);
background:linear-gradient(to left,#ffffff 0,#ffffff 20%,rgba(255, 255, 255, 0) 100%);
color:rgba(0, 0, 0, 0.54);
height:24px
}
@media screen and (min-width:1440px){
.profile-snippet .snippet-fade{
background:-webkit-linear-gradient(left,#f7f7f7 0,#f7f7f7 20%,rgba(247, 247, 247, 0) 100%);
background:linear-gradient(to left,#f7f7f7 0,#f7f7f7 20%,rgba(247, 247, 247, 0) 100%)
}
}
@media screen and (max-width:800px){
.blog-name{
margin-top:0
}
body.item-view .blog-name{
margin:0 48px
}
.centered-bottom{
padding:8px
}
body.item-view .centered-bottom{
padding:0
}
.page_body .centered{
padding:10px 0
}
body.item-view #header,body.item-view .widget.Header{
margin-right:0
}
body.collapsed-header .centered-top-container .blog-name{
display:block
}
body.collapsed-header .centered-top-container .widget.Header h1{
text-align:center
}
.widget.Header header{
padding:0
}
.widget.Header h1{
font-size:24px;
line-height:24px;
margin-bottom:13px
}
body.item-view .widget.Header h1{
text-align:center
}
body.item-view .widget.Header p{
text-align:center
}
.blog-name .widget.PageList{
padding:0
}
body.item-view .centered-top{
margin-bottom:5px
}
.search-action,.search-input{
margin-bottom:-8px
}
.search form{
margin-bottom:8px
}
body.item-view .subscribe-section-container{
margin:5px 0 0 0;
width:100%
}
#page_body.section div.widget.FeaturedPost,div.widget.PopularPosts{
padding:16px
}
div.widget.Blog .blog-posts .post-outer-container{
padding:16px
}
div.widget.Blog .blog-posts .post-outer-container .post-outer{
padding:0
}
.post:first-child{
margin:0
}
.post-body .snippet-thumbnail{
margin:0 3vw 3vw 0
}
.post-body .snippet-thumbnail img{
height:20vw;
width:20vw;
max-height:128px;
max-width:128px
}
div.widget.PopularPosts div.item-thumbnail{
margin:0 3vw 3vw 0
}
div.widget.PopularPosts div.item-thumbnail img{
height:20vw;
width:20vw;
max-height:88px;
max-width:88px
}
.post-title{
line-height:1
}
.post-title,.post-title a{
font-size:20px
}
#page_body.section div.widget.FeaturedPost h3 a{
font-size:22px
}
.mobile-ad .inline-ad{
display:block
}
.page_body.has-vertical-ads .vertical-ad-container,.page_body.has-vertical-ads .vertical-ad-container ins{
display:none
}
.page_body.has-vertical-ads .centered .centered-bottom,.page_body.has-vertical-ads .centered .centered-top{
display:block;
width:auto
}
div.post-filter-message div{
padding:8px 16px
}
}
@media screen and (min-width:1440px){
body{
position:relative
}
body.item-view .blog-name{
margin-left:48px
}
.page_body{
margin-left:284px
}
.search{
margin-left:0
}F
.search.focused{
width:100%
}
.sticky{
padding-left:284px
}
.hamburger-menu{
display:none
}
body.collapsed-header .page_body .centered-top-container{
padding-left:284px;
padding-right:0;
width:100%
}
body.collapsed-header .centered-top-container .search.focused{
width:100%
}
body.collapsed-header .centered-top-container .blog-name{
margin-left:0
}
body.collapsed-header.item-view .centered-top-container .search.focused{
width:calc(100% - 50px)
}
body.collapsed-header.item-view .centered-top-container .blog-name{
margin-left:40px
}
}

--></style>
<style id='template-skin-1' type='text/css'><!--
body#layout .hidden,
body#layout .invisible {
display: inherit;
}
body#layout .navigation {
display: none;
}
body#layout .page,
body#layout .sidebar_top,
body#layout .sidebar_bottom {
display: inline-block;
left: inherit;
position: relative;
vertical-align: top;
}
body#layout .page {
float: right;
margin-left: 20px;
width: 55%;
}
body#layout .sidebar-container {
float: right;
width: 40%;
}
body#layout .hamburger-menu {
display: none;
}
--></style>
<style>
    .bg-photo {background-image:url(https\:\/\/themes.googleusercontent.com\/image?id=L1lcAxxz0CLgsDzixEprHJ2F38TyEjCyE3RSAjynQDks0lT1BDc1OxXKaTEdLc89HPvdB11X9FDw);}
    
@media (max-width: 480px) { .bg-photo {background-image:url(https\:\/\/themes.googleusercontent.com\/image?id=L1lcAxxz0CLgsDzixEprHJ2F38TyEjCyE3RSAjynQDks0lT1BDc1OxXKaTEdLc89HPvdB11X9FDw&options=w480);}}
@media (max-width: 640px) and (min-width: 481px) { .bg-photo {background-image:url(https\:\/\/themes.googleusercontent.com\/image?id=L1lcAxxz0CLgsDzixEprHJ2F38TyEjCyE3RSAjynQDks0lT1BDc1OxXKaTEdLc89HPvdB11X9FDw&options=w640);}}
@media (max-width: 800px) and (min-width: 641px) { .bg-photo {background-image:url(https\:\/\/themes.googleusercontent.com\/image?id=L1lcAxxz0CLgsDzixEprHJ2F38TyEjCyE3RSAjynQDks0lT1BDc1OxXKaTEdLc89HPvdB11X9FDw&options=w800);}}
@media (max-width: 1200px) and (min-width: 801px) { .bg-photo {background-image:url(https\:\/\/themes.googleusercontent.com\/image?id=L1lcAxxz0CLgsDzixEprHJ2F38TyEjCyE3RSAjynQDks0lT1BDc1OxXKaTEdLc89HPvdB11X9FDw&options=w1200);}}
/* Last tag covers anything over one higher than the previous max-size cap. */
@media (min-width: 1201px) { .bg-photo {background-image:url(https\:\/\/themes.googleusercontent.com\/image?id=L1lcAxxz0CLgsDzixEprHJ2F38TyEjCyE3RSAjynQDks0lT1BDc1OxXKaTEdLc89HPvdB11X9FDw&options=w1600);}}
  </style>
<script async='async' src='https://www.gstatic.com/external_hosted/clipboardjs/clipboard.min.js'></script>
<link href='https://www.blogger.com/dyn-css/authorization.css?targetBlogID=5899759802104611399&amp;zx=cf522af3-da41-4f2f-bab0-e81bdfac7783' media='none' onload='if(media!=&#39;all&#39;)media=&#39;all&#39;' rel='stylesheet'/><noscript><link href='https://www.blogger.com/dyn-css/authorization.css?targetBlogID=5899759802104611399&amp;zx=cf522af3-da41-4f2f-bab0-e81bdfac7783' rel='stylesheet'/></noscript>
<meta name='google-adsense-platform-account' content='ca-host-pub-1556223355139109'/>
<meta name='google-adsense-platform-domain' content='blogspot.com'/>

</head>
<body class='item-view version-1-3-3 variant-indie_light'>
<a class='skip-navigation' href='#main' tabindex='0'>
Skip to main content
</a>
<div class='page'>
<div class='bg-photo-overlay'></div>
<div class='bg-photo-container'>
<div class='bg-photo'></div>
</div>
<div class='page_body'>
<div class='centered'>
<div class='centered-top-placeholder'></div>
<header class='centered-top-container' role='banner'>
<div class='centered-top'>
<a class='return_link' href='https://controlexxp.blogspot.com/?m=1'>
<button class='svg-icon-24-button back-button rtl-reversible-icon flat-icon-button ripple'>
<svg class='svg-icon-24'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_arrow_back_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
</button>
</a>
<div class='search'>
<button aria-label='Search' class='search-expand touch-icon-button'>
<div class='flat-icon-button ripple'>
<svg class='svg-icon-24 search-expand-icon'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_search_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
</div>
</button>
<div class='section' id='search_top' name='Search (Top)'><div class='widget BlogSearch' data-version='2' id='BlogSearch1'>
<h3 class='title'>
Search This Blog
</h3>
<div class='widget-content' role='search'>
<form action='https://controlexxp.blogspot.com/search' target='_top'>
<div class='search-input'>
<input aria-label='Search this blog' autocomplete='off' name='q' placeholder='Search this blog' value=''/>
</div>
<input class='search-action flat-button' type='submit' value='Search'/>
</form>
</div>
</div></div>
</div>
<div class='clearboth'></div>
<div class='blog-name container'>
<div class='container section' id='header' name='Header'><div class='widget Header' data-version='2' id='Header1'>
<div class='header-widget'>
<div>
<h1>
<a href='https://controlexxp.blogspot.com/?m=1'>
Control
</a>
</h1>
</div>
<p>
</p>
</div>
</div></div>
<nav role='navigation'>
<div class='clearboth no-items section' id='page_list_top' name='Page List (Top)'>
</div>
</nav>
</div>
</div>
</header>
<div>
<div class='vertical-ad-container no-items section' id='ads' name='Ads'>
</div>
<main class='centered-bottom' id='main' role='main' tabindex='-1'>
<div class='main section' id='page_body' name='Page Body'>
<div class='widget Blog' data-version='2' id='Blog1'>
<div class='blog-posts hfeed container'>
<article class='post-outer-container'>
<div class='post-outer'>
<div class='post'>
<script type='application/ld+json'>{
  "@context": "http://schema.org",
  "@type": "BlogPosting",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://controlexxp.blogspot.com/2023/09/xyz.html"
  },
  "headline": "xyz","description": "1012690A1261012651143122LUJUHT712SMPP|ok 1012690A12610126|KNG-PRO|51143 1033790A33710337|KNG-PRO|C0453 1048090A48010480|KNG-PRO|C7492 104839...","datePublished": "2023-09-15T11:02:00-07:00",
  "dateModified": "2023-10-24T23:27:31-07:00","image": {
    "@type": "ImageObject","url": "https://blogger.googleusercontent.com/img/b/U2hvZWJveA/AVvXsEgfMvYAhAbdHksiBA24JKmb2Tav6K0GviwztID3Cq4VpV96HaJfy0viIu8z1SSw_G9n5FQHZWSRao61M3e58ImahqBtr7LiOUS6m_w59IvDYwjmMcbq3fKW4JSbacqkbxTo8B90dWp0Cese92xfLMPe_tg11g/w1200/",
    "height": 348,
    "width": 1200},"publisher": {
    "@type": "Organization",
    "name": "Blogger",
    "logo": {
      "@type": "ImageObject",
      "url": "https://blogger.googleusercontent.com/img/b/U2hvZWJveA/AVvXsEgfMvYAhAbdHksiBA24JKmb2Tav6K0GviwztID3Cq4VpV96HaJfy0viIu8z1SSw_G9n5FQHZWSRao61M3e58ImahqBtr7LiOUS6m_w59IvDYwjmMcbq3fKW4JSbacqkbxTo8B90dWp0Cese92xfLMPe_tg11g/h60/",
      "width": 206,
      "height": 60
    }
  },"author": {
    "@type": "Person",
    "name": "KING CYBER"
  }
}</script>
<a name='3345527317414303050'></a>
<h3 class='post-title entry-title'>
xyz
</h3>
<div class='post-share-buttons post-share-buttons-top'>
<div class='byline post-share-buttons goog-inline-block'>
<div aria-owns='sharing-popup-Blog1-byline-3345527317414303050' class='sharing' data-title='xyz'>
<button aria-controls='sharing-popup-Blog1-byline-3345527317414303050' aria-label='Share' class='sharing-button touch-icon-button' id='sharing-button-Blog1-byline-3345527317414303050' role='button'>
<div class='flat-icon-button ripple'>
<svg class='svg-icon-24'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_share_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
</div>
</button>
<div class='share-buttons-container'>
<ul aria-hidden='true' aria-label='Share' class='share-buttons hidden' id='sharing-popup-Blog1-byline-3345527317414303050' role='menu'>
<li>
<span aria-label='Get link' class='sharing-platform-button sharing-element-link' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Get link'>
<svg class='svg-icon-24 touch-icon sharing-link'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_link_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Get link</span>
</span>
</li>
<li>
<span aria-label='Share to Facebook' class='sharing-platform-button sharing-element-facebook' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=facebook' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Share to Facebook'>
<svg class='svg-icon-24 touch-icon sharing-facebook'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_facebook_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Facebook</span>
</span>
</li>
<li>
<span aria-label='Share to Twitter' class='sharing-platform-button sharing-element-twitter' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=twitter' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Share to Twitter'>
<svg class='svg-icon-24 touch-icon sharing-twitter'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_twitter_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Twitter</span>
</span>
</li>
<li>
<span aria-label='Share to Pinterest' class='sharing-platform-button sharing-element-pinterest' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=pinterest' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Share to Pinterest'>
<svg class='svg-icon-24 touch-icon sharing-pinterest'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_pinterest_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Pinterest</span>
</span>
</li>
<li>
<span aria-label='Email' class='sharing-platform-button sharing-element-email' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=email' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Email'>
<svg class='svg-icon-24 touch-icon sharing-email'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_email_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Email</span>
</span>
</li>
<li aria-hidden='true' class='hidden'>
<span aria-label='Share to other apps' class='sharing-platform-button sharing-element-other' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Share to other apps'>
<svg class='svg-icon-24 touch-icon sharing-sharingOther'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_more_horiz_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Other Apps</span>
</span>
</li>
</ul>
</div>
</div>
</div>
</div>
<div class='post-header'>
<div class='post-header-line-1'>
<span class='byline post-timestamp'>
<meta content='https://controlexxp.blogspot.com/2023/09/xyz.html'/>
<a class='timestamp-link' href='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' rel='bookmark' title='permanent link'>
<time class='published' datetime='2023-09-15T11:02:00-07:00' title='2023-09-15T11:02:00-07:00'>
September 15, 2023
</time>
</a>
</span>
</div>
</div>
<div class='post-body entry-content float-container' id='post-body-3345527317414303050'>
<p>1012690A1261012651143122LUJUHT712SMPP|ok</p><p>1012690A12610126|KNG-PRO|51143</p><p>1033790A33710337|KNG-PRO|C0453</p><p>1048090A48010480|KNG-PRO|C7492</p><p>1048390A4831048385028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>1048390A48310483|KNG-PRO|85028</p><p>1024690A24610246C1064118PESIRFFREP6311SMPP|ok</p><p>Mamun</p><p>1035990A35910359|KNG-PRO|25539</p><p>1035990A3591035925539111RPAEUT64584812771SMPP|ok</p><p>Varma</p><p>1059090A5901059065654192TCONOM713FB04GFREP131SMPP|ok</p><p>1024690A24610246|KNG-PRO|55851</p><p>1027190A27110271|KNG-PRO|C8004</p><p>1024690A2461024655851201RAMIRF1911SMPP|ok</p><p>1027190A27110271C8004023GUAEUTFREP0811SMPP|ok</p><p>Nx</p><p>1087190A8711087101235141NUJDEWFC32C58C9587GFREP6811SMPP|ok</p><p>1025990A25910259C3191612GUADEWFREP2511SMPP|ok</p><p>1024690A24610246|KNG-PRO|C1064</p><p>1022790A2271022741339113GUAUHTYTRID5CE084BFBG6811SMPP|ok</p><p>1053190A53110531C2380112GUADEWYTRID7CD7755D261EGIKGQ011SMPP|ok</p><p>1007090A7010070C9373814NAJDEWFREP7511SMPP|ok</p><p>1007090A7010070|KNG-PRO|C9373</p><p>1022790A22710227|KNG-PRO|41339</p><p>1048090A48010480C7492229BEFUHTFREP0811SMPP|ok</p><p>1007090A7010070C9373814NAJDEWFREP7511SMPP|ok</p><p>1033790A33710337C0453011GUAEUT1911SMPP|ok</p><p>1024590A24510245|KNG-PRO|C8471</p><p>1024590A24510245C8471025PESEUTFREP5211SMPP|ok</p><p>1025690A2561025653557172RAMNOM454814836D53GFREP6811SMPP|ok</p><p>1025690A25610256|KNG-PRO|53557</p><p>Hasan</p><p>1021290A21210212C2231214PESNOM464BA8967D37BFA0DG448066921DIORDNA011SMPP|ok</p><p><br /></p><p>1021290A21210212|KNG-PRO|C2231</p><p>Salman Hossain&nbsp;</p><p>1019590A19510195C9574715NAJDEW35BA3911SMPP|ok</p><p>1019590A19510195|KNG-PRO|C9574</p><p>Sabbir 7d accpt 3oct</p><p>1026690A26610266C2365717GUANOM1911SMPP|ok</p><p>1026690A26610266|KNG-PRO|C2365</p><p>Rafsan 6exit</p><p>1034590A34510345|KNG-PRO|C2410</p><p>1034590A34510345C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>Riyad</p><p>1026590A2651026524405191PESEUT1911SMPP|expired</p><p>1026590A26510265|KNG-PRO|24405</p><p>25sep&nbsp;</p><p>1060190A60110601C5352104GUAIRFFREP7511SMPP|fuck</p><p>1060190A60110601|KNG-PRO|C5352</p><p>-----</p><p>Salman</p><p>1025090A25010250|KNG-PRO|K6440</p><p>1025090A25010250K6440619NAJNOM30120622771SMPP|ok</p><p>Pavel Khan</p><p><br /></p><p>PK</p><p>1011590A1151011513943281NAJEUT712SMPP|ok</p><p><br /></p><p>1011590A11510115|KNG-PRO|13943</p><p><br /></p><p>1020490A2041020435755103GUADEWYTRID7C725D9F8D9EG1911SMPP|ok</p><p><br /></p><p>1020490A20410204|KNG-PRO|35755</p><p><br /></p><p>1024090A2401024052525172RPAUHTFREP721SMPP|ok</p><p>1024090A24010240|KNG-PRO|52525</p><p>1033290A33210332C5131112NUJIRFFREP6311SMPP|ok</p><p>1033290A33210332|KNG-PRO|C5131</p><p><br /></p><p><br /></p><p>1026490A26410264C0163611PESIRF1911SMPP|ok</p><p>1026490A26410264|KNG-PRO|C0163</p><p>Pk</p><p>1024090A2401024080627161PESTASYTRID5A9FABB10080G1911SMPP|ok</p><p><br /></p><p dir="ltr">1024090A24010240|KNG-PRO|80627</p><p dir="ltr">&nbsp;</p><p dir="ltr"><br /></p><p dir="ltr">1020590A20510205C3111115LUJNOM1411SMPP|ok</p><p dir="ltr"><br /></p><p dir="ltr">1020590A20510205|KNG-PRO|C3111</p><p>Hridoy</p><p>1022190A22110221C8495116LUJUHTYTRID2D18CE0F3666G7211SMPP|ok</p><p>1022190A22110221|KNG-PRO|C8495</p><p>%-%-%-%-%-%-%-%-%-%-%%-%</p><p>Atif</p><p>2nd phn</p><p>1019590A1951019543441151NUJUHTYTRID05EA315D2FE3G1911SMPP|ok</p><p>1019590A19510195|KNG-PRO|43441</p><p>3rd phn</p><p>1053290A53210532C4470103GUAUHTYTRIDDD35F9D9C126G613101911SMPP|ok</p><p>1053290A53210532|KNG-PRO|C4470</p><p>4rd phn</p><p>1040090A4001040082343203YAMEUTFREP2511SMPP|ok</p><p>1040090A40010400|KNG-PRO|82343</p><p>-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;-&amp;</p><p>Sakib</p><p>1027190A27110271|KNG-PRO|01235</p><p>1027190A2711027101235141NUJDEWFC32C58C9587GFREP6811SMPP|ok</p><p><br /></p><p>1027190A27110271|KNG-PRO|01235</p><p>Dhru</p><p>1046090A46010460C0315215LUJDEWFREP7511SMPP|ok</p><p>1046090A46010460|KNG-PRO|C0315</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Jisanx30</p><p>Join Date : Oct/3</p><p>Days : 30days</p><p>Key : 1019990A1991019933255111GUAIRFYTRID55FCF15EB9C5G585001911SMPP|ok</p><p>1019990A19910199|KNG-PRO|33255</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Shahriarx10</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key : 1014190A14110141C6263001PESIRF1911SMPP|ok</p><p>1014190A14110141|KNG-PRO|C6263</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : inc3</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key :1024090A24010240C8170029TCOTASFREP0111SMPP|ok</p><p>1024090A24010240|KNG-PRO|C8170</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : arafatx30d</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key : 1033390A33310333C1102811NUJUHT3911SMPP|ok</p><p>1033390A33310333|KNG-PRO|C1102</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;AHSANx10KC</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key :1026890A26810268K6094815YAMIRF31235822FREP721SMPP|ok</p><p>1026890A26810268|KNG-PRO|K6094</p><p><br /></p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Bikkuux10In</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key :</p><p>1028690A2861028615623112BEFEUTFREP7511SMPP|ok</p><p>1028690A28610286|KNG-PRO|15623</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : salmanx10KC</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key :1021690A21610216C9574715NAJDEW35BA3911SMPP|ok</p><p>1021690A21610216|KNG-PRO|C9574</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : MarufCXL7</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key :</p><p>1051990A51910519C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>1051990A51910519|KNG-PRO|C2410</p><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Mosarufxntc</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : un/days</div><div><br /></div><div>Key:1027690A2761027602934091LUJDEW8811SMPP|ok</div><div><br /></div><div>1027690A27610276|KNG-PRO|02934</div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : JONYx10LC</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key:</div><div>1017490A17410174U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</div><div>1017490A17410174|KNG-PRO|U2563</div><div><br /></div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : AtifX30KC</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 30days</div><div><br /></div><div>Key:</div><div>1069690A69610696C9153517BEFEUT6812SMPP|ok</div><div><br /></div><div>1069690A69610696|KNG-PRO|C9153</div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : ornobx10CK</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key:</div><div><p>1028790A2871028744222231CEDEUTYTRIDDC0541AAE630G7211SMPP|ok</p><p>1028790A28710287|KNG-PRO|44222</p></div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : sabbirx7ZK</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key :</div><div>1028990A28910289C9053222GUADEWFREP7511SMPP|ok</div><div>1028990A28910289|KNG-PRO|C9053</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Xudun</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key :&nbsp;</div></div><div>1018290A1821018244844162RPADEW091SMPP|ok</div><div>1018290A18210182|KNG-PRO|44844</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : sdcard</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key</div></div><div>1037690A37610376C4024228PESUHT755BA3911SMPP|ok</div><div>1037690A37610376|KNG-PRO|C4024</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name :Mohin</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key</div></div><div>1043090A4301043072210261CEDIRFA0426ED750ABG88200FREP6811SMPP|ok</div><div>1043090A43010430|KNG-PRO|72210</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name :&nbsp;</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key:</div><div>1026090A26010260C0453011GUAEUT1911SMPP|ok</div><div>1026090A26010260|KNG-PRO|C0453</div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Tanvirx10HFD</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key</div><div>1026090A26010260C0453011GUAEUT1911SMPP|ok</div><div>1026090A26010260|KNG-PRO|C0453</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : sunil</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key:</div></div><div>1022890A2281022841339113GUAUHTYTRID5CE084BFBG6811SMPP|ok</div><div>1022890A22810228|KNG-PRO|41339</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : oppix7KCD</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key</div></div><div>1026490A26410264C5534212RAMUHTFREP7511SMPP|ok</div><div>1026490A26410264|KNG-PRO|C5534</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : indian</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key:</div></div><div>1083290A8321083274829191RPADEWFREP721SMPP|ok</div><div>1083290A83210832|KNG-PRO|74829</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Jafrinxnoorx7</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key:</div><div>1029490A29410294C2365717GUANOM1911SMPP|ok</div><div>1029490A29410294|KNG-PRO|C2365</div><div>2nd</div><div><br /></div><div>1028290A28210282C2365717GUANOM1911SMPP|ok</div><div>1028290A28210282|KNG-PRO|C2365</div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : omorx7kvb</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key</div><div>1014890A14810148C3331412NUJIRFFREP7511SMPP|ok</div><div>1014890A14810148|KNG-PRO|C3331</div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : salamx10kc</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 10days</div><div><br /></div><div>Key</div></div><div>1029090A2901029015355141GUANOM384BAB61FE58FB06BG148066921DIORDNA011SMPP|ok</div><div><br /></div><div>1029090A29010290|KNG-PRO|15355</div><div><br /></div><div><br /></div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : RakibxKC15</div><div>Join Date : Oct/3</div><div>Days : 15days</div><div><br /></div><div>Key</div></div><div>1024790A24710247C8123019RPATAS6111SMPP|ok</div><div>1024790A24710247| KNG-PRO | C8123</div><div>2nd</div><div>1062290A6221062224405191PESEUT1911SMPP|ok</div><div>1062290A62210622| KNG-PRO | 24405</div><div>1062290A6221062224405191PESEUT1911SMPP|ok</div><div>1062290A62210622|KNG-PRO|24405</div><div>1025090A25010250C8123019RPATAS6111SMPP|ok</div><div>1025090A25010250|KNG-PRO|C8123</div><div><div><div>----🔥----😮&#8205;💨----🔥----</div><div><br /></div><div>Name : Millerx7kc</div><div><br /></div><div>Join Date : Oct/3</div><div><br /></div><div>Days : 7days</div><div><br /></div><div>Key : 1024390A2431024321150282RPAUHT6811SMPP|ok</div></div><div>1024390A24310243|KNG-PRO|21150</div></div></div></div></div></div></div></div></div><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : sabbirx7kcxw</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>1024090A2401024014730182BEFEUT6811SMPP|ok</p><p><br /></p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : emonx7dxk</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>10AHS90AAHS10AHSU0553209GUADEWD370432E1093GFREP7511SMPP|ok</p><p>10AHS90AAHS10AHS|KNG-PRO|U0553</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : drubax7mxr</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>1035990A35910359C5131112NUJIRFFREP6311SMPP|ok</p><p>1035990A35910359|KNG-PRO|C5131</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Siyamxsheikhx7kdt</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>1038990A3891038954350062BEFTAS773F6D5GFREP0811SMPP|ok</p><p>1038990A38910389|KNG-PRO|54350</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : romanx7jgg</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>1014290A1421014204710291RAMIRF72SMPP|ok</p><p>1014290A14210142|KNG-PRO|04710</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Shawonx7JXR</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1026490A26410264C2365717GUANOM1911SMPP|ok</p><p>1026490A26410264|KNG-PRO|C2365</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Milonx7CK</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1026490A26410264C2365717GUANOM1911SMPP|ok</p><p>1026490A26410264|KNG-PRO|C2365</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : nayem7ck</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:1023090A2301023011532231GUANUS3911SMPP|ok</p><p>1023090A23010230|KNG-PRO|11532</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : nibir7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:1041890A41810418C9232814LUJEUTFREP0811SMPP|ok</p><p>1041890A41810418|KNG-PRO|C9232</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : foysalx7gkt</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:1031790A3171031763844172LUJDEW97BCA6579B41GFREP6811SMPP|ok</p><p>1031790A31710317|KNG-PRO|63844</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : MdAbdurRouf7xkfh</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024490A2441024415355141GUANOM384BAB61FE58FB06BG148066921DIORDNA011SMPP|ok</p><p>1024490A24410244|KNG-PRO|15355</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Rayhanx7hf</p><p>Join Date : Oct/3</p><p>Days : 7days</p><p>Key:</p><p>1026990A26910269C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>1026990A26910269|KNG-PRO|C2410</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : roki</p><p>Join Date : sep/27</p><p>Days : 10days</p><p>Key:</p><p>1021290A21210212C5472613LUJNOMYTRIDD28E2C0E8495G1911SMPP|ok</p><p>1021290A21210212|KNG-PRO|C5472</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : zxrxun</p><p>Join Date : Oct/27</p><p>Days : un/days</p><p>Key:</p><p>1026090A2601026044040082NAJUHTFREP7111SMPP|ok</p><p>1026090A26010260|KNG-PRO|44040</p><p>1026690A2661026602240122RAMDEWFREP0811SMPP|ok</p><p>1026690A26610266|KNG-PRO|02240</p><p>1026890A2681026853432261BEFDEWYTRID1EA4A419FAECGFREP0912SMPP|ok</p><p>1026890A26810268|KNG-PRO|53432</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : hasanx7zxrt</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1039990A39910399C0042317LUJIRF1911SMPP|ok</p><p>1039990A39910399|KNG-PRO|C0042</p><p>1030890A3081030881542203NUJIRFFREP7511SMPP|ok</p><p>1030890A30810308|KNG-PRO|81542</p><p>1028090A2801028061954141GUANOMCE35B919E4C2G6811SMPP|ok</p><p>1028090A28010280|KNG-PRO|61954</p><p>1025290A2521025214911181YAMUHT37516101BAC731733C8278G9621SMPP|ok</p><p>1025290A25210252|KNG-PRO|14911</p><p>1021290A2121021260906031NUJEUT273BAF2B08F2F6010G428066921DIORDNA011SMPP|ok</p><p>1021290A21210212|KNG-PRO|60906</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Risalatx7KC</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1025690A25610256C9315818GUAEUT1911SMPP|ok</p><p>1025690A25610256|KNG-PRO|C9315</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : MamunxUn/L</p><p>Join Date : Oct/4</p><p>Days : un/days</p><p>Key:</p><p>1029690A29610296|KNG-PRO|75731</p><p>1029690A2961029675731182LUJIRF89934B1BDAC5GFREP7511SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : forhadx7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024090A2401024034443262YAMIRFYTRID25BE2C6F4D93G1911SMPP|ok</p><p>1024090A24010240|KNG-PRO|34443</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Mamunxun</p><p>Join Date : Oct/4</p><p>Days : un/days</p><p>Key:</p><p>1030890A3081030873848122CEDUHT195BA3911SMPP|ok</p><p>1030890A30810308|KNG-PRO|73848</p><p>1023990A2391023963844172LUJDEW97BCA6579B41GFREP6811SMPP|ok</p><p>1023990A23910239|KNG-PRO|63844</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : salamx10xcl</p><p>Join Date : Oct/3</p><p>Days : 10days</p><p>Key:</p><p>1023590A2351023594233241RAMEUT3E4E6219846DGFREP091SMPP|ok</p><p>1023590A23510235|KNG-PRO|94233</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : saifx7blr</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1029390A2931029324453262YAMIRFYTRID67CCA6880C7FG1911SMPP|ok</p><p>1029390A29310293|KNG-PRO|24453</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Himanshux10ind</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1024990A2491024995359002PESDEW1911SMPP|ok</p><p>1024990A24910249|KNG-PRO|95359</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : jisanx10KC</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1045290A45210452|KNG-PRO|C3534</p><p>1045290A45210452C3534317NAJIRF712SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : sakibxun</p><p>Join Date : Oct/4</p><p>Days : un/days</p><p>Key:</p><p>1022290A2221022240521292TCOUHTFREP501SMPP|ok</p><p>1022290A22210222|KNG-PRO|40521</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ashikxcr7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024590A2451024550858142LUJNOMFREP721SMPP|ok</p><p>1024590A24510245|KNG-PRO|50858</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : salman7dck</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Keok</p><p>1022290A22210222C5325616LUJUHTYTRID44BD1201CG6811SMPP|ok</p><p>1022290A22210222|KNG-PRO|C5325</p><p>Name : ajayx10kf</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1046190A4611046160458111LUJEUTYTRIDD23D1EF4A41BG7211SMPP|ok</p><p>1046190A46110461|KNG-PRO|60458</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;Reseterx7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1036190A3611036161409013GUADEWFREP7511SMPP|ok</p><p>1036190A36110361|KNG-PRO|61409</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : shahriarx7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024990A2491024984043022GUAEUTYTRID5A9FABB10080G1911SMPP|ok</p><p>1024990A24910249|KNG-PRO|84043</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : saifxx10kc</p><p>Join Date : Oct/26</p><p>Days : 10days</p><p>Key:</p><p>1026890A26810268C7060112RAMDEWFREP721SMPP|ok</p><p>1026890A26810268|KNG-PRO|C7060</p><p>Name :&nbsp;HasanxRajax10ind</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1026090A2601026010843101GUAUHT0629215FD343GFREP1911SMPP|ok</p><p>1026090A26010260|KNG-PRO|10843</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : mitaz30ck</p><p>Join Date : Oct/4</p><p>Days : 30days</p><p>Key:1038790A3871038763157161RPAIRF487B9A3GFREP7111SMPP|ok</p><p>1038790A38710387|KNG-PRO|63157</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : shinchanx10</p><p>Join Date : Oct/6</p><p>Days : 10days</p><p>Key:</p><p>1014790A1471014751921131PESIRFBBAB559CC6FCGSOEGAENIL0411SMPP|ok</p><p>1014790A14710147|KNG-PRO|51921</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rkgx10kc</p><p>Join Date : Oct/4</p><p>Days : 10days</p><p>Key:</p><p>1027190A27110271C1355215YAMIRFFREP2511SMPP|ok</p><p>1027190A27110271|KNG-PRO|C1355</p><p><br /></p><p>Name : tonux7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1025390A25310253K6440619NAJNOM30120622771SMPP|ok</p><p>1025390A25310253|KNG-PRO|K6440</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : kabbox7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1018490A18410184C8385115YAMUHT092SMPP|ok</p><p>1018490A18410184|KNG-PRO|C8385</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : jahanemux7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1026090A26010260C9315818GUAEUT1911SMPP|ok</p><p>1026090A26010260|KNG-PRO|C9315</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : hridoymiax15kc</p><p>Join Date : Oct/4</p><p>Days : days</p><p>Key:</p><p>1034390A3431034305719112VONNOMYTRID494AE7032FFAG7211SMPP|ok</p><p>1034390A34310343|KNG-PRO|05719</p><p><br /></p><p>Name : nazmulx7kc</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1024490A2441024414632181YAMNOMFREP892SMPP|ok</p><p>1024490A24410244|KNG-PRO|14632</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : hridoyx7kcy</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1025190A2511025195521103NUJIRF1911SMPP|ok</p><p>1025190A25110251|KNG-PRO|95521</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : xyzcrx7</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1073890A73810738C7023511PESIRFFREP2511SMPP|ok</p><p>1073890A73810738|KNG-PRO|C7023</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : tuhinx7xck</p><p>Join Date : Oct/4</p><p>Days : 7days</p><p>Key:</p><p>1039190A3911039194846191NUJNOM040488526811SMPP|ok</p><p>1039190A39110391|KNG-PRO|94846</p><p><br /></p><p><br /></p><p>Name : Asfafulx7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1029090A29010290C9315818GUAEUT1911SMPP|ok</p><p>1029090A29010290|KNG-PRO|C9315</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : sakibXkd10k</p><p>Join Date : Oct/5</p><p>Days : 10days</p><p>Key:</p><p>1021090A2101021073629172NUJEUTYTRIDF470E01560AEG1911SMPP|ok</p><p>1021090A21010210|KNG-PRO|73629</p><p>1028190A28110281C2203913GUAUHTFREP7511SMPP|ok</p><p>1028190A28110281|KNG-PRO|C2203</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : salmanx7kcm</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1042690A42610426C6335012NUJIRFFREP6311SMPP|ok</p><p>1042690A42610426|KNG-PRO|C6335</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : forhadxcc7</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1026890A2681026832125061YAMNOM3356958BAEDCBD73E0C4FG10000921DIORDNA1011SMPP|ok</p><p>1026890A26810268|KNG-PRO|32125</p><p><br /></p><p>Name :&nbsp;topux7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1019890A1981019895731162RPADEW091SMPP|ok</p><p>1019890A19810198|KNG-PRO|95731</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : habibx7kd</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1026390A2631026344222231CEDEUTYTRIDDC0541AAE630G7211SMPP|ok</p><p>1026390A26310263|KNG-PRO|44222</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Abubaker10ck</p><p>Join Date : Oct/5</p><p>Days : 10days</p><p>Key:1035990A35910359W5282007VONUHTF88A040GFREP131SMPP|ok</p><p>1035990A35910359|KNG-PRO|W5282</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : onvrcx7</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1031690A31610316C8195218GUAEUTFREP7511SMPP|ok</p><p>1031690A31610316|KNG-PRO|C8195</p><p><br /></p><p>Name : Riazx7x9</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1033190A33110331C6333818NUJUHT3C3F549D678AGIKGQ011SMPP|ok</p><p>1033190A33110331|KNG-PRO|C6333</p><p>----------------------------------------</p><p><br /></p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Khalidx7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1026590A2651026585627152TCONOMYTRID4A17375G7211SMPP|ok</p><p>1026590A26510265|KNG-PRO|85627</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rajax10ind</p><p>Join Date : Oct/5</p><p>Days : 10days</p><p>Key:</p><p>1006890A6810068C9373814NAJDEWFREP7511SMPP|ok</p><p>1006890A6810068|KNG-PRO|C9373</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : mxk7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1026890A2681026801857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1026890A26810268|KNG-PRO|01857</p><p>Name : mostakimx7knb</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1098790A98710987|KNG-PRO|85028</p><p>1098790A9871098785028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rafixn15kc</p><p>Join Date : Oct/5</p><p>Days : 15days</p><p>Key:</p><p>1020690A2061020601544162GUATAS946BA3934B34662CDG921DIORDNA011SMPP|ok</p><p>1020690A20610206|KNG-PRO|01544</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : kmxpix7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1032590A3251032594349151GUAEUTFREP7511SMPP|ok</p><p>1032590A32510325|KNG-PRO|94349</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : soyednurx7xokc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1025490A25410254|KNG-PRO|85028</p><p>1025590A25510255|KNG-PRO|44407</p><p>1025390A25310253|KNG-PRO|13111</p><p>1022490A22410224|KNG-PRO|12824</p><p>[&#8730;] 1022490A2241022412824172CEDNUS407FA01GFREP681SMPP|ok</p><p>[&#8730;] 1025490A2541025485028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>[&#8730;] 1025590A2551025544407003CEDIRFYTRID6875E1C8FA19GFREP7511SMPP|ok</p><p>[&#8730;] 1025390A2531025313111092NUJEUTC169E08GFREP0812SMPP|ok</p><p>Name : xkmr7xk</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1031690A3161031685028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>1031690A31610316|KNG-PRO|85028</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : syedx7kcb</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1047790A47710477|KNG-PRO|72210</p><p>1047790A4771047772210261CEDIRFA0426ED750ABG88200FREP6811SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>ktx7kc</p><p>Join Date : Oct/5</p><p>Days : 7days</p><p>Key:</p><p>1045790A4571045730524152LUJEUTA32F9178CEEEGFREP6811SMPP|ok</p><p>1045790A45710457|KNG-PRO|30524</p><p>----🔥----😮&#8205;💨----🔥---</p><p>Name : indx10kc</p><p>Join Date : Oct/5</p><p>Days : 10days</p><p>Key:</p><p>1026690A2661026663332271RAMUHT6811SMPP|ok</p><p>1026690A26610266|KNG-PRO|63332</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>NamSMPridoyx10kc2100</p><p>Join Date : Oct/6</p><p>Days : 10days</p><p>Key:</p><p>1</p><p>1026590A26510265C3224114GUAIRFFREP2511SMPP|ok</p><p>1026590A26510265|KNG-PRO|C3224</p><p>2</p><p>1033290A33210332C2365717GUANOM1911SMPP|ok</p><p>1033290A33210332|KNG-PRO|C2365</p><p>3</p><p>1015190A1511015101857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1015190A15110151|KNG-PRO|01857</p><p>4</p><p>1065590A6551065501857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1065590A65510655|KNG-PRO|01857</p><p>5</p><p>1034490A3441034420213011RAMIRF092SMPP|ok</p><p>1034490A34410344|KNG-PRO|20213</p><p>6</p><p>1030590A30510305C2365717GUANOM1911SMPP|ok</p><p>1030590A30510305|KNG-PRO|C2365</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : aminulx10kc</p><p>Join Date : Oct/6</p><p>Days : 10days</p><p>Key:</p><p>1027390A27310273U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>1027390A27310273|KNG-PRO|U2563</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : sharifulx7kc</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1020590A20510205C9052222GUADEWYTRIDFDBBD7557B24G415101911SMPP|ok</p><p>1020590A20510205|KNG-PRO|C9052</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : shakibxkcspider</p><p>Join Date : Oct/9</p><p>Days : x/days</p><p>Key:</p><p>1025590A25510255|KNG-PRO|C3523</p><p>1025590A25510255C3523004VONUHT2235F04B1C11GFREP6811SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : mamun</p><p>Join Date : Oct/5</p><p>Days : days</p><p>Key:</p><p>1026490A26410264C6542228YAMNOM0D1F8CC5DFD9GFREP0911SMPP|ok</p><p>1026490A26410264|KNG-PRO|C6542</p><p>1069190A6911069124407141RAMEUTFREP21SMPP|ok</p><p>1069190A69110691|KNG-PRO|24407</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : sushil</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1036590A36510365C1064118PESIRFFREP6311SMPP|ok</p><p>1036590A36510365|KNG-PRO|C1064</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : rajx7kc</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1043790A4371043735243101LUJNOM53DA5ED1AAC8GIKGQ011SMPP|ok</p><p>1043790A43710437|KNG-PRO|35243</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : alifx7kc</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1014490A14410144|KNG-PRO|62531</p><p>1014490A1441014462531232TCODEWFREP351SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : tomx10kc</p><p>Join Date : Oct/9</p><p>Days : 10days</p><p>Key:</p><p>1051290A51210512C5131112NUJIRFFREP6311SMPP|ok</p><p>1051290A51210512|KNG-PRO|C5131</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rodiulx7kc</p><p>Join Date : Oct/9</p><p>Days : 7days</p><p>Key:</p><p>1026790A2671026722751013GUAUHT593BA6D0F69E8E8B6G921DIORDNA011SMPP|ok</p><p>1026790A26710267|KNG-PRO|22751</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : rohitx7kc</p><p>Join Date : Oct/10</p><p>Days : 7days</p><p>Key:</p><p>1042990A4291042971240071YAMEUT6111SMPP|ok</p><p>1042990A42910429|KNG-PRO|71240</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : Tohinx7kc</p><p>Join Date : Oct/11</p><p>Days : 7days</p><p>Key:</p><p>1019590A1951019522518152LUJNOM39272412111SMPP|ok</p><p>1019590A19510195|KNG-PRO|22518</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : torikulx7kc</p><p>Join Date : Oct/12</p><p>Days : 7days</p><p>Key:</p><p>1023790A23710237|KNG-PRO|U2563</p><p>1023790A23710237U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;Habibx7KC</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1021390A21310213|KNG-PRO|73629</p><p>1021390A2131021373629172NUJEUTYTRIDF470E01560AEG1911SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : rayhanx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1026590A26510265|KNG-PRO|C9315</p><p>1026590A26510265C9315818GUAEUT1911SMPP|ok</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : insanx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1052490A5241052491657101RPANOM80BF6D7B5639GFREP0911SMPP|ok</p><p>1052490A52410524|KNG-PRO|91657</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : mdrakibx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1023490A2341023402535102PESDEWFREP0911SMPP|ok</p><p>1023490A23410234|KNG-PRO|02535</p><p>&nbsp;----🔥----😮&#8205;💨----🔥----</p><p>Name : angal priya</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1013390A13310133C9162716LUJUHTYTRID44BD1201CG6811SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : unx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1013390A13310133C9162716LUJUHTYTRID44BD1201CG6811SMPP|ok</p><p>1013390A13310133|KNG-PRO|C9162</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1032090A3201032082323241PESDEWFREP2511SMPP|ok</p><p>1032090A32010320|KNG-PRO|82323</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rumpax7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1020590A20510205C5303717PESDEWYTRID75D2441346ECG7211SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : arafatx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1024290A24210242|KNG-PRO|C5241</p><p>1024290A24210242C5241223RAMIRFYTRID58E6EA100E44G1911SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ahad</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1027990A27910279C2365717GUANOM1911SMPP|ok</p><p>1027990A27910279|KNG-PRO|C2365</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : jj</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1053190A53110531C2380112GUADEWYTRID7CD7755D261EGIKGQ011SMPP|ok</p><p>1053190A53110531|KNG-PRO|C2380</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : sehjad</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1032390A3231032375655103GUADEWYTRID7C725D9F8D9EG1911SMPP|ok</p><p>1032390A32310323|KNG-PRO|75655</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ronyx7kc</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1024890A2481024814911181YAMUHT37516101BAC731733C8278G9621SMPP|ok</p><p>1024890A24810248|KNG-PRO|14911</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : un</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1026790A2671026745612252PESNOM444BAA5D78983035CG921DIORDNA011SMPP|ok</p><p>1026790A26710267|KNG-PRO|45612</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : fahim</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1038190A3811038160435131CEDEUTFREP0811SMPP|ok</p><p>1038190A38110381|KNG-PRO|60435</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ornobx10kc</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1027890A27810278|KNG-PRO|83530</p><p>1027890A278102788353017GUANOM27752822771SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;Chand 10 days</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1025990A25910259C3191612GUADEWFREP2511SMPP|ok</p><p>1025990A25910259|KNG-PRO|C3191</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Arman 10 days</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1021990A2191021950712211TCOEUTFREP501SMPP|ok</p><p>1021990A21910219|KNG-PRO|50712</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : abu</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1029990A29910299C8195218GUAEUTFREP7511SMPP|ok</p><p>1029990A29910299|KNG-PRO|C8195</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : emranx7</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1039890A3981039892350191NUJNOM7FDF633ADC49GFREP0911SMPP|ok</p><p>1039890A39810398|KNG-PRO|92350</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ganja khor</p><p>Join Date : Oct/15</p><p>Days : days</p><p>Key:</p><p>1026590A26510265|KNG-PRO|C5352</p><p>1026590A26510265C5352104GUAIRFFREP7511SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rakib</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1027890A2781027872549152YAMUHT85250201BAE64F36A5401DG90000831DIORDNA871SMPP|ok</p><p>1027890A27810278|KNG-PRO|72549</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : dines</p><p>Join Date : Oct/15</p><p>Days : 10days</p><p>Key:</p><p>1027190A2711027112010062VONIRFFREP721SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : hasib islam</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1028490A28410284U5220017PESUHTE5895E8E0F01G1911SMPP|ok</p><p>1028490A28410284|KNG-PRO|U5220</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : fahimxnox</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p><br /></p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : rafsanxun</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1039390A3931039351453292LUJIRF1411SMPP|ok</p><p>1025290A2521025212241111GUAIRFYTRID3B3461F3E63CG1911SMPP|ok</p><p>1026790A2671026712241111GUAIRFYTRID3B3461F3E63CG1911SMPP|ok</p><p>1025290A25210252|KNG-PRO|12241</p><p>1039390A39310393|KNG-PRO|51453</p><p>1026790A26710267|KNG-PRO|12241</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : robinmixx7</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1015290A1521015234601161RAMUHTFREP21SMPP|ok</p><p>1015290A15210152|KNG-PRO|34601</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : mamunxfire</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1030990A30910309C8195218GUAEUTFREP7511SMPP|ok</p><p>1030990A30910309|KNG-PRO|C8195</p><p>1059090A5901059065654192TCONOM713FB04GFREP131SMPP|ok</p><p>1027190A2711027112010062VONIRFFREP721SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p><br /></p><p>Name : maruf</p><p>Join Date : Oct/15</p><p>Days : days</p><p>Key:</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Ml hasan</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1090790A90710907C1395025PESEUT1911SMPP|ok</p><p>1090790A90710907|KNG-PRO|C1395</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Mst</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1027790A2771027750110092GUAEUT1911SMPP|ok</p><p>1027690A2761027650110092GUAEUT1911SMPP|ok</p><p>1026890A2681026821708012LUJIRF89934B1BDAC5GFREP7511SMPP|ok</p><p>1027890A2781027850110092GUAEUT1911SMPP|ok</p><p>1028090A2801028005951111GUAIRFFREP2511SMPP|ok</p><p>1027190A2711027150110092GUAEUT1911SMPP|ok</p><p>1026890A26810268|KNG-PRO|21708</p><p>1028090A28010280|KNG-PRO|05951</p><p>1027190A27110271|KNG-PRO|50110</p><p>1027890A27810278|KNG-PRO|50110</p><p>1027690A27610276|KNG-PRO|50110</p><p>1027790A27710277|KNG-PRO|50110</p><p>Tarif2</p><p><br /></p><p>1025390A25310253C7032024PESNOM1911SMPP|ok</p><p><br /></p><p>1025390A25310253|KNG-PRO|C7032</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : atif</p><p>Join Date : Oct/15</p><p>Days : 9days</p><p>Key:</p><p>1050890A5081050883230112NAJIRFFREP7112SMPP|ok</p><p>1050890A50810508|KNG-PRO|83230</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : torikul</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1010290A10210102|KNG-PRO|95709</p><p>1010290A1021010295709131TCODEW49862744BQ28714651191SMPP|ok</p><p><br /></p><p>tarif 1</p><p>1040090A4001040054911292TCOUHTFREP501SMPP|ok</p><p>1040090A40010400|KNG-PRO|54911</p><p>Hasan 4</p><p>1027490A2741027441625182GUANOME4466B858E64GFREP6811SMPP|ok</p><p><br /></p><p>1027490A27410274|KNG-PRO|41625</p><p>Hasan 5</p><p>1025290A25210252U0390905NUJNOM29983601BA428323F7F0F1G8721SMPP|ok</p><p><br /></p><p><br /></p><p>1025290A25210252|KNG-PRO|U0390</p><p>Hasan 3</p><p>1027590A2751027505951111GUAIRFFREP2511SMPP|ok</p><p>1027590A27510275|KNG-PRO|05951</p><p>hasan 2</p><p>501101039990A3991039950110092GUAEUT1911SMPP|ok</p><p>1019890A19810198|KNG-PRO|C4470</p><p>1019890A19810198C4470103GUAUHTYTRIDDD35F9D9C126G613101911SMPP|ok</p><p>1039990A39910399|KNG-PRO|50110</p><p>hasan 1</p><p>1030890A30810308C2452106PESDEWFREP7511SMPP|ok</p><p><br /></p><p>1030890A30810308|KNG-PRO|C2452</p><p>Hasan 7&nbsp;</p><p>1023990A23910239|KNG-PRO|73353</p><p>1023990A2391023973353101CEDIRFFREP291SMPP|ok</p><p>Hasan 6</p><p>1023890A23810238|KNG-PRO|84007</p><p>1023890A2381023884007101GUAUHT413BA3911SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : nitu</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1031190A31110311C9232814LUJEUTFREP0811SMPP|ok</p><p>1031190A31110311|KNG-PRO|C9232</p><p>----🔥----😮&#8205;💨----🔥----</p><p>1059090A5901059065654192TCONOM713FB04GFREP131SMPP|ok</p><p>1027190A2711027112010062VONIRFFREP721SMPP|ok</p><p>Name : hasib islam</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1024690A24610246C4175021GUAEUTBB399011EA68G1911SMPP|ok</p><p><br /></p><p>1024690A24610246|KNG-PRO|C4175</p><p>1059090A59010590|KNG-PRO|65654</p><p>1059090A5901059065654192TCONOM713FB04GFREP131SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : tonmoyx7</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1022890A22810228C9410718PESIRFFREP2511SMPP|ok</p><p>1022890A22810228|KNG-PRO|C9410</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;Khandoker Ahasanul Haque</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1024390A2431024302910231TCOUHT68D003598739G28200FREP6811SMPP|ok</p><p>1024390A24310243|KNG-PRO|02910</p><p>1027190A27110271|KNG-PRO|12010</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : tanvir loskar</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1023290A2321023202021101RPANOMFREP721SMPP|ok</p><p>1023290A23210232|KNG-PRO|02021</p><p>1027190A2711027112010062VONIRFFREP721SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : forhad hossain</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1027890A27810278C2365717GUANOM1911SMPP|ok</p><p>1027890A27810278|KNG-PRO|C2365</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : marufx7</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1020590A20510205C5303717PESDEWYTRID75D2441346ECG7211SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : imran fzk</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1022890A22810228U1451315NUJNOM27926201BA9A794AE460E6G11000831DIORDNA471SMPP|ok</p><p>1022890A22810228|KNG-PRO|U1451</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : roki khan</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1026090A26010260C2012917PESUHTYTRID6788BB5D2DB7G1911SMPP|ok</p><p>1026090A26010260|KNG-PRO|C2012</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : anas</p><p>Join Date : Oct/15</p><p>Days : 7days</p><p>Key:</p><p>1037190A3711037171150201RPANOM51FC9D87D1EEGFREP6811SMPP|ok</p><p>1037190A37110371|KNG-PRO|71150</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : mohin</p><p>Join Date : Oct/16</p><p>Days : 7days</p><p>Key:</p><p>1023490A23410234|KNG-PRO|25022</p><p>1023490A2341023425022161RAMUHT21944891771SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Chandan&nbsp;</p><p>Join Date : Oct/16</p><p>Days : 7days</p><p>Key:</p><p>1025290A25210252|KNG-PRO|40021</p><p>1026190A26110261|KNG-PRO|C6464</p><p>1026190A26110261C6464818PESIRFFREP7511SMPP|ok</p><p>1044090A44010440|KNG-PRO|50001</p><p>1044090A4401044050001191NUJNOMAF0F3DCE1C91GFREP0911SMPP|ok</p><p>1025290A2521025240021111YAMDEW6811SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ci</p><p>Join Date : Oct/16</p><p>Days : 7days</p><p>Key:</p><p>1029590A2951029594349151GUAEUTFREP7511SMPP|ok</p><p>1029590A29510295|KNG-PRO|94349</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : faizal</p><p>Join Date : Oct/16</p><p>Days : 10days</p><p>Key:</p><p>1026690A26610266|KNG-PRO|U2563</p><p>1026690A26610266U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>Shomir&nbsp; sir er key</p><p>1027790A2771027753233121GUATAS328147523111SMPP|ok</p><p><br /></p><p>1027790A27710277|KNG-PRO|53233</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : Sakibx7</p><p>Join Date : Oct/18</p><p>Days : 7days</p><p>Key:</p><p>1025190A2511025101857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1025190A25110251|KNG-PRO|01857</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/18</p><p>Days : 7days</p><p>Key:</p><p>1018490A18410184C4071811GUAEUTYTRID34F577C29C08G1911SMPP|ok</p><p>1018490A18410184|KNG-PRO|C4071</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/18</p><p>Days : 7days</p><p>Key:</p><p>1023090A2301023051504111VONIRFFREP7112SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : ariyan</p><p>Join Date : Oct/19</p><p>Days : 7days</p><p>Key:</p><p>1030390A3031030385923211VONIRFFREP7511SMPP|ok</p><p>1030390A30310303|KNG-PRO|85923</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : efratx7</p><p>Join Date : Oct/20</p><p>Days : 7days</p><p>Key:</p><p>1076590A7651076501235141NUJDEWFC32C58C9587GFREP6811SMPP|ok</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name : xxx</p><p>Join Date : Oct/20</p><p>Days : 7days</p><p>Key:</p><p>1022890A22810228C7234718TCONUSFREP2511SMPP|ok</p><p>1022890A22810228|KNG-PRO|C7234</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/</p><p>Days : days</p><p>Key:</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/</p><p>Days : days</p><p>Key:</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/</p><p>Days : days</p><p>Key:</p><p>----🔥----😮&#8205;💨----🔥----</p><p>Name :&nbsp;</p><p>Join Date : Oct/</p><p>Days : days</p><p>Key:</p><p>Dulavai</p><p>1027090A27010270C4261613GUAUHTYTRID3B69BB6AE714G6811SMPP|ok</p><p>1027090A27010270|KNG-PRO|C4261</p><p>1026890A26810268K6094815YAMIRF31235822FREP721SMPP|ok</p><p>1026890A26810268|KNG-PRO|K6094</p><p>Rafsan user</p><p>1034590A34510345C2410514YAMUHTYTRID8C2A31802G7211SMPP|expired</p><p>1034590A34510345|KNG-PRO|C2410</p><p>🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥</p><p><br /></p><p>Date 😎28/ September 2023</p><p><br /></p><p>Sakib ref</p><p><br /></p><p>1032590A32510325C9315818GUAEUT1911SMPP|ok</p><p><br /></p><p>1032590A32510325|KNG-PRO|C9315</p><p><br /></p><p><br /></p><p><br /></p><p>Costello&nbsp;</p><p><br /></p><p>1024090A2401024003741152GUAIRF8F521085D3CAGFREP7211SMPP|ok</p><p><br /></p><p>1024090A24010240|KNG-PRO|03741</p><p><br /></p><p><br /></p><p><br /></p><p>Shinchan nuhara</p><p><br /></p><p>1028590A2851028524708142RAMIRFYTRIDA1BBCF1470A8G7211SMPP|ok</p><p><br /></p><p>1028590A28510285|KNG-PRO|24708</p><p><br /></p><p>Forhad&nbsp;</p><p><br /></p><p>1022490A2241022424708142RAMIRFYTRIDA1BBCF1470A8G7211SMPP|ok</p><p><br /></p><p>1022490A22410224|KNG-PRO|24708</p><p><br /></p><p>Miraz</p><p><br /></p><p>1013190A1311013104710291RAMIRF72SMPP|ok</p><p><br /></p><p>1013190A13110131|KNG-PRO|04710</p><p><br /></p><p>Onik</p><p><br /></p><p>1027590A2751027590411102BEFNOM0609DA91E20CGFREP0911SMPP|ok</p><p>1027590A27510275|KNG-PRO|90411</p><p><br /></p><p>sakib 10d</p><p><br /></p><p>1022290A2221022283230112NAJIRFFREP7112SMPP|ok</p><p><br /></p><p>1022290A22210222|KNG-PRO|83230</p><p><br /></p><p><br /></p><p><br /></p><p><br /></p><p><br /></p><p>1023190A23110231C5131112NUJIRFFREP6311SMPP|ok</p><p><br /></p><p>1023190A23110231|KNG-PRO|C5131</p><p><br /></p><p>Date 😎29/ September 2023</p><p><br /></p><p>1036390A3631036343633161YAMNOM2EVB4UXXF525ABA82447242FREP0911SMPP|ok</p><p><br /></p><p>1036390A36310363|KNG-PRO|43633</p><p><br /></p><p>1027790A27710277C1193123RAMIRFYTRIDCBCB558EA0F7G6811SMPP|ok</p><p><br /></p><p>1027790A27710277|KNG-PRO|C1193</p><p><br /></p><p>1017890A1781017840856151BEFEUT092SMPP|ok</p><p><br /></p><p>1017890A17810178|KNG-PRO|40856</p><p><br /></p><p>1031790A31710317|KNG-PRO|71802</p><p><br /></p><p>1031790A3171031771802202NUJEUTYTRID3E127485A8F9G013101911SMPP|ok</p><p><br /></p><p><br /></p><p><br /></p><p>1023990A2391023980627161PESTASYTRID5A9FABB10080G1911SMPP|ok</p><p><br /></p><p>1023990A23910239|KNG-PRO|80627</p><p><br /></p><p>Shohid</p><p><br /></p><p>1025390A25310253C0430312GUAEUT9FA24C3C6E70GFREP0811SMPP|ok</p><p><br /></p><p>1025390A25310253|KNG-PRO|C0430</p><p><br /></p><p>Shaheen</p><p><br /></p><p>1033790A3371033721740161GUADEW166071723111SMPP|ok</p><p><br /></p><p>1033790A33710337|KNG-PRO|21740</p><p><br /></p><p><br /></p><p><br /></p><p>1042590A4251042591657101RPANOM80BF6D7B5639GFREP0911SMPP|ok</p><p><br /></p><p>1042590A42510425|KNG-PRO|91657</p><p><br /></p><p><br /></p><p><br /></p><p><br /></p><p><br /></p><p>Sabina 10days</p><p><br /></p><p>1035790A3571035735237102TCOUHT1C9D67D65584GFREP7511SMPP|ok</p><p><br /></p><p>1035790A35710357|KNG-PRO|35237</p><p><br /></p><p><br /></p><p><br /></p><p>1029790A29710297U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok&nbsp;</p><p><br /></p><p>1029790A29710297|KNG-PRO|U2563</p><p><br /></p><p><br /></p><p><br /></p><p>1032590A32510325U2114403GUAUHTE60BBC0E1595GFREP6811SMPP|ok</p><p><br /></p><p>1032590A32510325|KNG-PRO|U2114</p><p><br /></p><p><br /></p><p><br /></p><p>Forhad</p><p><br /></p><p>1031190A3111031135144151NUJUHT0981F57BE0C0GFREP1911SMPP|ok</p><p><br /></p><p>1031190A31110311|KNG-PRO|35144</p><p><br /></p><p><br /></p><p><br /></p><p>1031790A31710317|KNG-PRO|C4470</p><p><br /></p><p>1031790A31710317C4470103GUAUHTYTRIDDD35F9D9C126G613101911SMPP|ok</p><p><br /></p><p>MD LION ALI</p><p><br /></p><p>1014590A1451014534601161RAMUHTFREP21SMPP|ok&nbsp;</p><p><br /></p><p>1014590A14510145|KNG-PRO|34601</p><p><br /></p><p>Redoy</p><p><br /></p><p>1020390A20310203C9052222GUADEWYTRIDFDBBD7557B24G415101911SMPP|ok</p><p><br /></p><p>1020390A20310203|KNG-PRO|C9052</p><p><br /></p><p>Miraj</p><p><br /></p><p>1059790A59710597C9315818GUAEUT1911SMPP|ok</p><p><br /></p><p>1059790A59710597|KNG-PRO|C9315</p><p><br /></p><p><br /></p><p><br /></p><p>Sabina 1021990A2191021983412172CEDEUTYTRIDB60A7B1E9238G879001911SMPP|ok</p><p><br /></p><p>1021990A21910219|KNG-PRO|83412</p><p><br /></p><p><br /></p><p><br /></p><p>Aditya</p><p><br /></p><p>1023890A2381023884043022GUAEUTYTRID5A9FABB10080G1911SMPP|ok</p><p><br /></p><p>1023890A23810238|KNG-PRO|84043</p><p><br /></p><p><br /></p><p><br /></p><p>Ornob 2nd</p><p><br /></p><p>1018890A1881018825730262LUJDEW6A4727B20082GSAE201VSUXEN201SMPP|ok</p><p><br /></p><p>1018890A18810188|KNG-PRO|25730</p><p><br /></p><p><br /></p><p><br /></p><p>1031090A31010310J0304306NUJNOMFREP5211SMPP|ok</p><p><br /></p><p>1031090A31010310|KNG-PRO|J0304</p><p><br /></p><p><br /></p><p><br /></p><p>Date 😎30/ September 2023</p><p><br /></p><p>Albert&nbsp;</p><p><br /></p><p>1033490A3341033455521041CEDEUTFREP0812SMPP|ok</p><p><br /></p><p>1033490A33410334|KNG-PRO|55521</p><p><br /></p><p><br /></p><p><br /></p><p>1030190A3011030135755103GUADEWYTRID7C725D9F8D9EG1911SMPP|ok</p><p><br /></p><p>1030190A30110301|KNG-PRO|35755</p><p><br /></p><p>Sabbir</p><p><br /></p><p>1024390A2431024320404091RPANOMB9FA437GFREP7111SMPP|ok</p><p><br /></p><p>1024390A24310243|KNG-PRO|20404</p><p><br /></p><p>Salman</p><p><br /></p><p>1049290A4921049224229151NUJUHT506147523111SMPP|ok</p><p><br /></p><p>1049290A49210492|KNG-PRO|24229</p><p><br /></p><p>Minhaz</p><p><br /></p><p>1025290A2521025201857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p><br /></p><p>1025290A25210252|KNG-PRO|01857</p><p><br /></p><p><br /></p><p><br /></p><p>1022690A22610226C9023811GUANOM091SMPP|ok</p><p><br /></p><p>1034590A34510345U7052603GUAUHTE60BBC0E1595GFREP6811SMPP|ok</p><p><br /></p><p>1022690A22610226|KNG-PRO|C9023</p><p><br /></p><p>1034590A34510345|KNG-PRO|U7052</p><p><br /></p><p><br /></p><p>1021790A2171021744833052GUAIRFYTRID5A9FABB10080G1911SMPP|ok</p><p>1021790A21710217|KNG-PRO|44833</p><p><br /></p><p>1028790A28710287C2365717GUANOM1911SMPP|ok</p><p>1028790A28710287|KNG-PRO|C2365</p><p>Rana</p><p>1036390A36310363C7562309NUJUHTFREP7511SMPP|ok</p><p>1036390A36310363|KNG-PRO|C7562</p><p>Rakib</p><p>1048090A48010480C7492229BEFUHTFREP0811SMPP|ok</p><p>1048090A48010480|KNG-PRO|C7492</p><p><br /></p><p>Siyam Sheikh&nbsp;</p><p>1028590A2851028595521103NUJIRF1911SMPP|ok</p><p>1028590A28510285|KNG-PRO|95521</p><p>Sabbir 10days</p><p>1025390A2531025384531082RAMEUT9DE3CDC964DEGIKGQ011SMPP|ok</p><p>1025390A25310253|KNG-PRO|84531</p><p>Gaja khor 10days&nbsp;</p><p>1028190A2811028195521103NUJIRF1911SMPP|ok</p><p>1028190A28110281|KNG-PRO|95521|ok</p><p>Likhon 7days</p><p>1021990A21910219C5325616LUJUHTYTRID44BD1201CG6811SMPP|ok</p><p>1021990A21910219|KNG-PRO|C5325</p><p>Goru chor 10 days</p><p>1028690A2861028695521103NUJIRF1911SMPP|ok</p><p>1028690A28610286|KNG-PRO|95521</p><p>Abrar 7 din</p><p>1004690A461004685028142RPANOMYTRID5230CB7806A1GFREP7511SMPP|ok</p><p>1004690A4610046|KNG-PRO|85028</p><p>Nanir khali gor 10days</p><p>1021890A21810218C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>1021890A21810218|KNG-PRO|C2410</p><p>Atif 10d</p><p>1014090A14010140|KNG-PRO|03710</p><p>1014090A1401014003710281NAJNOMFREP21SMPP|ok</p><p>U1003590A3510035|KNG-PRO|C1395</p><p>1003590A3510035C1395025PESEUT1911SMPP|ok</p><p><br /></p><p>1028390A2831028394551141NUJEUT092SMPP|ok</p><p>1028390A28310283|KNG-PRO|94551</p><p><br /></p><p>Rifat 7days</p><p>1027090A2701027011952241PESDEWFREP2511SMPP|ok</p><p>1027090A27010270|KNG-PRO|11952</p><p>7days</p><p>1014590A1451014534948151NUJEUT711SMPP|ok</p><p>1014590A14510145|KNG-PRO|34948</p><p><br /></p><p>7days</p><p>1025090A2501025001857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1025090A25010250|KNG-PRO|01857</p><p>7days&nbsp;</p><p>1030990A30910309C9232814LUJEUTFREP0811SMPP|fuck</p><p>1030990A30910309|KNG-PRO|C9232</p><p>10days</p><p>1024890A24810248K8393515PESNOM71317322771SMPP|ok</p><p>1024890A24810248|KNG-PRO|K8393</p><p>7days</p><p>1029890A2981029805951111GUAIRFFREP2511SMPP|ok</p><p>1029890A29810298|KNG-PRO|05951</p><p>7days</p><p>1025390A2531025301857112VONNOM9595139BAA7F09EE7FB1FG04000921DIORDNA6311SMPP|ok</p><p>1025390A25310253|KNG-PRO|01857</p><p>Indian anas</p><p>1060590A6051060542057131GUAIRF21SMPP|fuck</p><p>1060590A60510605|KNG-PRO|42057</p><p>7days</p><p>Oct 1 🔥🔥🔥🔥🔥🔥🔥🔥</p><p>1029790A2971029785923211VONIRFFREP7511SMPP|ok</p><p>1029790A29710297|KNG-PRO|85923</p><p>7days&nbsp;</p><p>1026890A2681026841219141NUJDEW5006CE2AD8D2GFREP6811SMPP|ok</p><p>1026890A26810268|KNG-PRO|41219</p><p>7days&nbsp;</p><p>1027390A27310273|KNG-PRO|C3391</p><p>1027390A27310273C3391517RPAIRF1911SMPP|ok</p><p>Shishir</p><p>1032890A3281032844317172YAMDEW711SMPP|ok</p><p>1032890A32810328|KNG-PRO|44317</p><p>Reseter 7days</p><p>1029990A2991029985923211VONIRFFREP7511SMPP|ok</p><p>1029990A29910299|KNG-PRO|85923</p><p>Abdullah 7days</p><p>1024190A2411024190508041PESDEW0292609BAAEC500552FE6G6721SMPP|ok</p><p>1024190A24110241|KNG-PRO|90508</p><p>7days</p><p>1030590A30510305C0133618GUAEUT1911SMPP|ok&nbsp;</p><p>1030590A30510305|KNG-PRO|C0133</p><p>PK</p><p>Ali 7 din</p><p>1014690A1461014631808141BEFEUTYTRID16706ED1C3A6G669001911SMPP|ok</p><p>1014690A14610146|KNG-PRO|31808</p><p><br /></p><p>1023390A23310233C2215013LUJNOMYTRID4015E474EG6811SMPP|ok</p><p>1023390A23310233|KNG-PRO|C2215</p><p><br /></p><p>1027790A2771027773833121YAMIRF1911SMPP|ok</p><p>1027790A27710277|KNG-PRO|73833</p><p><br /></p><p>1028590A2851028533041213YAMDEWIKGQ011SMPP|ok</p><p>1028590A28510285|KNG-PRO|33041</p><p><br /></p><p>1022590A22510225C9145818PESDEW55CBA20GFREP7111SMPP|ok</p><p>1022590A22510225|KNG-PRO|C9145</p><p><br /></p><p>Riyad 10days</p><p>1027390A2731027324405191PESEUT1911SMPP|ok</p><p>1027390A27310273|KNG-PRO|24405</p><p>Tonu 10days</p><p>1026190A2611026163623011LUJEUT89934B1BDAC5GFREP7511SMPP|ok</p><p>1026190A26110261|KNG-PRO|63623</p><p>Emon 7days</p><p>1029390A2931029372348121NUJNOMFREP7511SMPP|ok</p><p>1029390A29310293|KNG-PRO|72348</p><p>7days</p><p>1032190A3211032154101201BEFIRFFREP7511SMPP|ok</p><p>1039490A39410394U7402517YAMNUS0349BDC786C2GFREP0811SMPP|ok</p><p>1032190A32110321|KNG-PRO|54101</p><p>1039490A39410394|KNG-PRO|U7402</p><p>10days</p><p>1024090A24010240|KNG-PRO|50205</p><p>1024090A2401024050205091GUATAS563BAB84803AB0CB4G921DIORDNA011SMPP|ok</p><p>1030090A30010300|KNG-PRO|83530</p><p>1030090A300103008353017GUANOM27752822771SMPP|ok</p><p>7days</p><p>1057290A5721057242524132RAMUHT443202422311SMPP|ok</p><p>1057290A57210572|KNG-PRO|42524</p><p>7days</p><p>Naeem Sharif:</p><p>1023990A23910239C2060103GUAUHT7211SMPP|ok</p><p>Raz Vaw:</p><p>1031990A31910319|KNG-PRO|21033</p><p><br /></p><p>Raz:</p><p>1023990A23910239|KNG-PRO|C2060</p><p><br /></p><p>Nahid Khan:</p><p>1025490A25410254|KNG-PRO|11952</p><p><br /></p><p>MD Rabbi:</p><p>1015190A15110151|KNG-PRO|80627</p><p>MD Rabbi:</p><p>1015190A1511015180627161PESTASYTRID5A9FABB10080G1911SMPP|ok</p><p><br /></p><p>Raz Vaw:</p><p>1031990A3191031921033142GUAUHTAC19649818F7GIKGQ011SMPP|ok</p><p><br /></p><p>Ahad Airtel:</p><p>1025490A2541025411952241PESDEWFREP2511SMPP|ok</p><p>Pk</p><p>Robin 1 mas</p><p>1020990A20910209C8443916NUJEUTYTRIDB6DF857E1A06G6811SMPP|ok</p><p>1020990A20910209|KNG-PRO|C8443</p><p>7days</p><p>1020190A20110201W5052008VONIRFFREP351SMPP|ok</p><p>1020190A20110201|KNG-PRO|W5052</p><p>7days</p><p>1004090A4010040|KNG-PRO|32513</p><p>1004090A401004032513012LUJIRF102BA97E9373D1BF8G921DIORDNA011SMPP|ok</p><p>15days</p><p>1025990A25910259U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>1025990A25910259|KNG-PRO|U2563</p><p><br /></p><p>1046690A4661046694551141NUJEUT092SMPP|ok</p><p>1046690A46610466|KNG-PRO|94551</p><p>India 7days</p><p>1090690A9061090601830291LUJEUTE80E285E3B0EGFREP6811SMPP|ok</p><p>1090690A90610906|KNG-PRO|01830</p><p>7days&nbsp;</p><p>1027890A27810278C3331412NUJIRFFREP7511SMPP|ok</p><p>1027890A27810278|KNG-PRO|C3331</p><p>2 October😎 2023</p><p>7days</p><p>1023490A2341023471341151VONEUTE191519F70E2GFREP6811SMPP|ok</p><p>1023490A23410234|KNG-PRO|71341</p><p><br /></p><p>7days</p><p>1028490A28410284U9091614LUJEUT0FC857007FF2G1911SMPP|ok</p><p>1024990A2491024905340011RAMIRF773F6D5GFREP0811SMPP|ok</p><p>1028490A28410284|KNG-PRO|U9091</p><p>1024990A24910249|KNG-PRO|05340</p><p>India 10days</p><p>1050990A5091050910059192NUJDEWFREP0811SMPP|ok</p><p>1050990A50910509|KNG-PRO|10059</p><p>Robin</p><p>1035090A35010350U7052603GUAUHTE60BBC0E1595GFREP6811SMPP|ok</p><p>1035090A35010350|KNG-PRO|U7052</p><p>Ashik 7days</p><p>1024590A24510245C0241211NUJUHTFREP721SMPP|ok</p><p>1024590A24510245|KNG-PRO|C0241</p><p>7days</p><p>1026990A26910269C2410514YAMUHTYTRID8C2A31802G7211SMPP|ok</p><p>1024590A24510245|KNG-PRO|C0241</p><p>Me</p><p>1028190A2811028180906112GUANOM582BCBAF9D61GFREP1911SMPP|ok</p><p>1028190A28110281|KNG-PRO|80906</p><p>7days</p><p>1065690A6561065654330182BEFEUT1CWD1UDK14VCSBA931400620912SMPP|ok</p><p>1035390A3531035395151151LUJTAS204684FD141BGIKGQ011SMPP|ok</p><p>1035390A35310353|KNG-PRO|95151</p><p>1065690A65610656|KNG-PRO|54330</p><p>Indian 10days</p><p>1040190A4011040121532191CEDNOM6811SMPP|ok</p><p>1040190A40110401|KNG-PRO|21532</p><p>pk</p><p>1034490A3441034454350062BEFTAS773F6D5GFREP0811SMPP|ok</p><p>1032290A32210322|KNG-PRO|95656</p><p>1032290A3221032295656101BEFIRFYTRID5CDB3454A0DFG7211SMPP|ok</p><p>1034490A34410344|KNG-PRO|54350</p><p><br /></p><p>Pk 1 mas</p><p>1021790A21710217C9493815PESEUT1911SMPP|ok</p><p>1021790A21710217|KNG-PRO|C9493</p><p>pk 7 din</p><p>1025690A2561025681427101RPANOM80BF6D7B5639GFREP0911SMPP|ok</p><p>1025690A25610256|KNG-PRO|81427</p><p>Pk 7 din</p><p>1028290A2821028232125061YAMNOM3356958BAEDCBD73E0C4FG10000921DIORDNA1011SMPP|ok</p><p>1028290A28210282|KNG-PRO|32125</p><p>7days</p><p>1021790A2171021795445131PESDEW751BAYTRID5EC5147EC7DCG921DIORDNA011SMPP|ok</p><p>1021790A21710217|KNG-PRO|95445</p><p><br /></p><p>Pk 10 din</p><p>1026490A2641026473833121YAMIRF1911SMPP|ok</p><p>1026490A26410264|KNG-PRO|73833</p><p><br /></p><p>India 10 din</p><p>1020790A2071020703811241TCODEW091SMPP|ok</p><p>1020790A20710207|KNG-PRO|03811</p><p>pk 7 din&nbsp;</p><p>1033790A3371033754559152YAMDEW86C3D25981C7G6811SMPP|ok</p><p>1033790A33710337|KNG-PRO|54559</p><p>India 10 din</p><p>=1029390A29310293C1371917PESUHT1911SMPP|ok</p><p>1029390A29310293|KNG-PRO|C1371</p><p>7 din</p><p>1022990A2291022915355141GUANOM384BAB61FE58FB06BG148066921DIORDNA011SMPP|ok</p><p>1022990A22910229|KNG-PRO|15355</p><p>india 10 din&nbsp;</p><p>1020690A20610206C6231219BEFUHT3E4E6219846DGFREP091SMPP|ok</p><p>1020690A20610206|KNG-PRO|C6231</p><p>India 10d</p><p>1036590A36510365C6422617GUANOMFREP6311SMPP|ok</p><p>1036590A36510365|KNG-PRO|C6422</p><p>7 din</p><p>1040190A4011040124708142RAMIRFYTRIDA1BBCF1470A8G7211SMPP|ok</p><p>1040190A40110401|KNG-PRO|24708</p><p><br /></p><p>7 din</p><p>1040990A4091040913427111NUJTAS091SMPP|ok</p><p>1040990A40910409|KNG-PRO|13427</p><p><br /></p><p><br /></p><p>Sushil indian 10 din</p><p>1036590A36510365C6422617GUANOMFREP6311SMPP|ok</p><p>1036590A36510365|KNG-PRO|C6422</p><p><br /></p><p>7 din&nbsp;</p><p>1025090A2501025002240122RAMDEWFREP0811SMPP|ok</p><p>1025090A25010250|KNG-PRO|02240</p><p>7 din&nbsp;</p><p>1027190A27110271C9033411GUAEUT6811SMPP|ok</p><p>1027190A27110271|KNG-PRO|C9033</p><p>7 din</p><p>1024790A2471024795709131TCODEW49862744BQ28714651191SMPP|ok</p><p>1024790A24710247|KNG-PRO|95709</p><p>10 din&nbsp;</p><p>1047090A4701047023934032RAMUHTYTRIDDB1DF96F5824G6811SMPP|ok</p><p>1047090A47010470|KNG-PRO|23934</p><p>7 din&nbsp;</p><p>1026790A2671026702021101RPANOMFREP721SMPP|ok</p><p>1026790A26710267|KNG-PRO|02021</p><p>7 din</p><p>1016890A1681016850513221LUJIRFFREP381SMPP|ok</p><p>1016890A16810168|KNG-PRO|50513</p><p>7 Din&nbsp;</p><p>1023990A23910239C9382512NUJUHT7211SMPP|ok</p><p>1023990A23910239|KNG-PRO|C9382</p><p><br /></p><p>7 din</p><p>1044090A4401044013943281NAJEUT712SMPP|ok</p><p>1044090A44010440|KNG-PRO|13943</p><p>7 din</p><p>1026990A26910269U2563917RAMEUT8583079BA24E9A1BC44D8G13000921DIORDNA9411SMPP|ok</p><p>1026990A26910269|KNG-PRO|U2563</p><p>Nida</p><p>1045390A4531045322910021GUATASYTRIDB48E7833B30EG1911SMPP|ok</p><p>1045390A45310453|KNG-PRO|22910</p><p><br /></p><p><br /></p><p>1023690A2361023684119141PESUHT8711BA69433C438F91G921DIORDNA011SMPP|ok</p><p><br /></p><p>1023690A23610236|KNG-PRO|84119</p><p><br /></p><p>1023690A2361023690130072PESDEW594BA69433C438F91G921DIORDNA011SMPP|ok</p><p><br /></p><p>1023690A23610236|KNG-PRO|90130</p><p><br /></p><p>1022690A2261022653847121PESEUTYTRID0F35F8D233A7G1911SMPP|ok</p><p><br /></p><p>1022690A22610226|KNG-PRO|53847</p><p><br /></p><p><br /></p><p>1025690A25610256C4071811GUAEUTYTRID34F577C29C08G1911SMPP|ok</p><p><br /></p><p>1025690A25610256|KNG-PRO|C4071</p>
</div>
<div class='post-bottom'>
<div class='post-footer float-container'>
<div class='post-footer-line post-footer-line-1'>
</div>
<div class='post-footer-line post-footer-line-2'>
</div>
<div class='post-footer-line post-footer-line-3'>
</div>
</div>
<div class='post-share-buttons post-share-buttons-bottom invisible'>
<div class='byline post-share-buttons goog-inline-block'>
<div aria-owns='sharing-popup-Blog1-byline-3345527317414303050' class='sharing' data-title='xyz'>
<button aria-controls='sharing-popup-Blog1-byline-3345527317414303050' aria-label='Share' class='sharing-button touch-icon-button' id='sharing-button-Blog1-byline-3345527317414303050' role='button'>
<div class='flat-icon-button ripple'>
<svg class='svg-icon-24'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_share_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
</div>
</button>
<div class='share-buttons-container'>
<ul aria-hidden='true' aria-label='Share' class='share-buttons hidden' id='sharing-popup-Blog1-byline-3345527317414303050' role='menu'>
<li>
<span aria-label='Get link' class='sharing-platform-button sharing-element-link' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Get link'>
<svg class='svg-icon-24 touch-icon sharing-link'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_link_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Get link</span>
</span>
</li>
<li>
<span aria-label='Share to Facebook' class='sharing-platform-button sharing-element-facebook' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=facebook' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Share to Facebook'>
<svg class='svg-icon-24 touch-icon sharing-facebook'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_facebook_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Facebook</span>
</span>
</li>
<li>
<span aria-label='Share to Twitter' class='sharing-platform-button sharing-element-twitter' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=twitter' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Share to Twitter'>
<svg class='svg-icon-24 touch-icon sharing-twitter'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_twitter_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Twitter</span>
</span>
</li>
<li>
<span aria-label='Share to Pinterest' class='sharing-platform-button sharing-element-pinterest' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=pinterest' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Share to Pinterest'>
<svg class='svg-icon-24 touch-icon sharing-pinterest'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_pinterest_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Pinterest</span>
</span>
</li>
<li>
<span aria-label='Email' class='sharing-platform-button sharing-element-email' data-href='https://www.blogger.com/share-post.g?blogID=5899759802104611399&postID=3345527317414303050&target=email' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Email'>
<svg class='svg-icon-24 touch-icon sharing-email'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_24_email_dark' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Email</span>
</span>
</li>
<li aria-hidden='true' class='hidden'>
<span aria-label='Share to other apps' class='sharing-platform-button sharing-element-other' data-url='https://controlexxp.blogspot.com/2023/09/xyz.html?m=1' role='menuitem' tabindex='-1' title='Share to other apps'>
<svg class='svg-icon-24 touch-icon sharing-sharingOther'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_more_horiz_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<span class='platform-sharing-text'>Other Apps</span>
</span>
</li>
</ul>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<section class='comments threaded' data-embed='true' data-num-comments='31' id='comments'>
<a name='comments'></a>
<h3 class='title'>Comments</h3>
<div class='comments-content'>
<script async='async' src='' type='text/javascript'></script>
<script type='text/javascript'>(function(){var aa=function(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}},n="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;a[b]=c.value;return a},ba=function(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("Cannot find global object");
},r=ba(this),t=function(a,b){if(b)a:{var c=r;a=a.split(".");for(var e=0;e<a.length-1;e++){var d=a[e];if(!(d in c))break a;c=c[d]}a=a[a.length-1];e=c[a];b=b(e);b!=e&&null!=b&&n(c,a,{configurable:!0,writable:!0,value:b})}};
t("Symbol",function(a){if(a)return a;var b=function(g,p){this.$jscomp$symbol$id_=g;n(this,"description",{configurable:!0,writable:!0,value:p})};b.prototype.toString=function(){return this.$jscomp$symbol$id_};var c="jscomp_symbol_"+(1E9*Math.random()>>>0)+"_",e=0,d=function(g){if(this instanceof d)throw new TypeError("Symbol is not a constructor");return new b(c+(g||"")+"_"+e++,g)};return d});
t("Symbol.iterator",function(a){if(a)return a;a=Symbol("Symbol.iterator");for(var b="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),c=0;c<b.length;c++){var e=r[b[c]];"function"===typeof e&&"function"!=typeof e.prototype[a]&&n(e.prototype,a,{configurable:!0,writable:!0,value:function(){return ca(aa(this))}})}return a});
var ca=function(a){a={next:a};a[Symbol.iterator]=function(){return this};return a},da=function(a,b){a instanceof String&&(a+="");var c=0,e=!1,d={next:function(){if(!e&&c<a.length){var g=c++;return{value:b(g,a[g]),done:!1}}e=!0;return{done:!0,value:void 0}}};d[Symbol.iterator]=function(){return d};return d};t("Array.prototype.entries",function(a){return a?a:function(){return da(this,function(b,c){return[b,c]})}});/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
var u=this||self,w=function(a,b){function c(){}c.prototype=b.prototype;a.superClass_=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.base=function(e,d,g){for(var p=Array(arguments.length-2),q=2;q<arguments.length;q++)p[q-2]=arguments[q];return b.prototype[d].apply(e,p)}},x=function(a){return a};function y(a,b){if(Error.captureStackTrace)Error.captureStackTrace(this,y);else{var c=Error().stack;c&&(this.stack=c)}a&&(this.message=String(a));void 0!==b&&(this.cause=b)}w(y,Error);y.prototype.name="CustomError";function z(a,b){a=a.split("%s");for(var c="",e=a.length-1,d=0;d<e;d++)c+=a[d]+(d<b.length?b[d]:"%s");y.call(this,c+a[e])}w(z,y);z.prototype.name="AssertionError";var A=function(a,b){throw new z("Failure"+(a?": "+a:""),Array.prototype.slice.call(arguments,1));};var E;var G=function(a,b){if(b!==F)throw Error("TrustedResourceUrl is not meant to be built directly");this.privateDoNotAccessOrElseTrustedResourceUrlWrappedValue_=a};G.prototype.toString=function(){return this.privateDoNotAccessOrElseTrustedResourceUrlWrappedValue_+""};var F={};var I=function(a){if(H!==H)throw Error("SafeUrl is not meant to be built directly");this.privateDoNotAccessOrElseSafeUrlWrappedValue_=a};I.prototype.toString=function(){return this.privateDoNotAccessOrElseSafeUrlWrappedValue_.toString()};var H={};new I("about:invalid#zClosurez");new I("about:blank");var J={},K=function(){if(J!==J)throw Error("SafeStyle is not meant to be built directly");this.privateDoNotAccessOrElseSafeStyleWrappedValue_=""};K.prototype.toString=function(){return this.privateDoNotAccessOrElseSafeStyleWrappedValue_.toString()};new K;var L={},M=function(){if(L!==L)throw Error("SafeStyleSheet is not meant to be built directly");this.privateDoNotAccessOrElseSafeStyleSheetWrappedValue_=""};M.prototype.toString=function(){return this.privateDoNotAccessOrElseSafeStyleSheetWrappedValue_.toString()};new M;var N={},P=function(){var a=u.trustedTypes&&u.trustedTypes.emptyHTML||"";if(N!==N)throw Error("SafeHtml is not meant to be built directly");this.privateDoNotAccessOrElseSafeHtmlWrappedValue_=a};P.prototype.toString=function(){return this.privateDoNotAccessOrElseSafeHtmlWrappedValue_.toString()};new P;var ea=function(){var a=document;var b="SCRIPT";"application/xhtml+xml"===a.contentType&&(b=b.toLowerCase());return a.createElement(b)};var Q=function(a,b){this.name=a;this.value=b};Q.prototype.toString=function(){return this.name};var R=new Q("OFF",Infinity),ka=new Q("WARNING",900),la=new Q("CONFIG",700),ma=function(){this.capacity_=0;this.clear()},S;ma.prototype.clear=function(){this.buffer_=Array(this.capacity_);this.curIndex_=-1;this.isFull_=!1};var T=function(a,b,c){this.reset(a||R,b,c,void 0,void 0)};T.prototype.reset=function(){};
var na=function(a,b){this.level=null;this.handlers=[];this.parent=(void 0===b?null:b)||null;this.children=[];this.logger={getName:function(){return a}}},oa=function(a){if(a.level)return a.level;if(a.parent)return oa(a.parent);A("Root logger has no level set.");return R},pa=function(a,b){for(;a;)a.handlers.forEach(function(c){c(b)}),a=a.parent},qa=function(){this.entries={};var a=new na("");a.level=la;this.entries[""]=a},U,V=function(a,b){var c=a.entries[b];if(c)return c;c=V(a,b.slice(0,Math.max(b.lastIndexOf("."),
0)));var e=new na(b,c);a.entries[b]=e;c.children.push(e);return e},W=function(){U||(U=new qa);return U};/*

 SPDX-License-Identifier: Apache-2.0
*/
var ra=[],sa=function(a){var b;if(b=V(W(),"safevalues").logger){var c="A URL with content '"+a+"' was sanitized away.",e=ka;if(a=b)if(a=b&&e){a=e.value;var d=b?oa(V(W(),b.getName())):R;a=a>=d.value}if(a){e=e||R;a=V(W(),b.getName());"function"===typeof c&&(c=c());S||(S=new ma);d=S;b=b.getName();if(0<d.capacity_){var g=(d.curIndex_+1)%d.capacity_;d.curIndex_=g;d.isFull_?(d=d.buffer_[g],d.reset(e,c,b),b=d):(d.isFull_=g==d.capacity_-1,b=d.buffer_[g]=new T(e,c,b))}else b=new T(e,c,b);pa(a,b)}}};
-1===ra.indexOf(sa)&&ra.push(sa);function ta(a,b){if(b instanceof G&&b.constructor===G)b=b.privateDoNotAccessOrElseTrustedResourceUrlWrappedValue_;else{var c=typeof b;A("expected object of type TrustedResourceUrl, got '%s' of type %s",b,"object"!=c?c:b?Array.isArray(b)?"array":c:"null");b="type_error:TrustedResourceUrl"}a.src=b;var e,d;(e=(b=null==(d=(e=(a.ownerDocument&&a.ownerDocument.defaultView||window).document).querySelector)?void 0:d.call(e,"script[nonce]"))?b.nonce||b.getAttribute("nonce")||"":"")&&a.setAttribute("nonce",
e)};function ua(a){a=null===a?"null":void 0===a?"undefined":a;if("string"!==typeof a)throw Error("Expected a string");if(void 0===E){var b=null;var c=u.trustedTypes;if(c&&c.createPolicy)try{b=c.createPolicy("goog#html",{createHTML:x,createScript:x,createScriptURL:x})}catch(e){u.console&&u.console.error(e.message)}E=b}a=(b=E)?b.createScriptURL(a):a;return new G(a,F)};var va=function(a,b,c){var e=null;a&&0<a.length&&(e=parseInt(a[a.length-1].timestamp,10)+1);var d=null,g=null,p=void 0,q=null,B=(window.location.hash||"#").substring(1),fa,ha;/^comment-form_/.test(B)?fa=B.substring(13):/^c[0-9]+$/.test(B)&&(ha=B.substring(1));var wa={id:c.postId,data:a,loadNext:function(l){if(e){var k=c.feed+"?alt=json&v=2&orderby=published&reverse=false&max-results=50";e&&(k+="&published-min="+(new Date(e)).toISOString());window.bloggercomments=function(C){e=null;var v=[];if(C&&
C.feed&&C.feed.entry)for(var f,ia=0;f=C.feed.entry[ia];ia++){var m={},h=/blog-(\d+).post-(\d+)/.exec(f.id.$t);m.id=h?h[2]:null;a:{h=void 0;var ja=f&&(f.content&&f.content.$t||f.summary&&f.summary.$t)||"";if(f&&f.gd$extendedProperty)for(h in f.gd$extendedProperty)if("blogger.contentRemoved"==f.gd$extendedProperty[h].name){h='<span class="deleted-comment">'+ja+"</span>";break a}h=ja}m.body=h;m.timestamp=Date.parse(f.published.$t)+"";f.author&&f.author.constructor===Array&&(h=f.author[0])&&(m.author=
{name:h.name?h.name.$t:void 0,profileUrl:h.uri?h.uri.$t:void 0,avatarUrl:h.gd$image?h.gd$image.src:void 0});f.link&&(f.link[2]&&(m.link=m.permalink=f.link[2].href),f.link[3]&&(h=/.*comments\/default\/(\d+)\?.*/.exec(f.link[3].href))&&h[1]&&(m.parentId=h[1]));m.deleteclass="item-control blog-admin";if(f.gd$extendedProperty)for(var D in f.gd$extendedProperty)"blogger.itemClass"==f.gd$extendedProperty[D].name?m.deleteclass+=" "+f.gd$extendedProperty[D].value:"blogger.displayTime"==f.gd$extendedProperty[D].name&&
(m.displayTime=f.gd$extendedProperty[D].value);v.push(m)}e=50>v.length?null:parseInt(v[v.length-1].timestamp,10)+1;l(v);window.bloggercomments=null};var O=ea();O.type="text/javascript";ta(O,ua(k+"&callback=bloggercomments"));document.getElementsByTagName("head")[0].appendChild(O)}},hasMore:function(){return!!e},getMeta:function(l,k){return"iswriter"==l?k.author&&k.author.name==c.authorName&&k.author.profileUrl==c.authorUrl?"true":"":"deletelink"==l?c.baseUri+"/delete-comment.g?blogID="+c.blogId+"&postID="+
k.id:"deleteclass"==l?k.deleteclass:""},onReply:function(l,k){null==d&&(d=document.getElementById("comment-editor"),null!=d&&(q=d.style.height,d.style.display="block",g=d.src.split("#")));d&&l&&l!==p&&(document.getElementById(k).insertBefore(d,null),k=g[0]+(l?"&parentID="+l:""),g[1]&&(k=k+"#"+g[1]),d.src=k,d.style.height=q||d.style.height,p=l,d.removeAttribute("data-resized"),d.dispatchEvent(new Event("iframeMoved")))},rendered:!0,initComment:ha,initReplyThread:fa,config:{maxDepth:c.maxThreadDepth},
messages:b};a=function(){if(window.goog&&window.goog.comments){var l=document.getElementById("comment-holder");window.goog.comments.render(l,wa)}};window.goog&&window.goog.comments?a():(window.goog=window.goog||{},window.goog.comments=window.goog.comments||{},window.goog.comments.loadQueue=window.goog.comments.loadQueue||[],window.goog.comments.loadQueue.push(a))},X=["blogger","widgets","blog","initThreadedComments"],Y=u;X[0]in Y||"undefined"==typeof Y.execScript||Y.execScript("var "+X[0]);
for(var Z;X.length&&(Z=X.shift());)X.length||void 0===va?Y=Y[Z]&&Y[Z]!==Object.prototype[Z]?Y[Z]:Y[Z]={}:Y[Z]=va;}).call(this);
</script>
<script type='text/javascript'>
    blogger.widgets.blog.initThreadedComments(
        null,
        null,
        {});
  </script>
<div id='comment-holder'>
<div class="comment-thread toplevel-thread"><ol id="top-ra"><li class="comment" id="c8231254969717140993"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11527560372885422365" rel="nofollow">TANVIR HERE</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695542012146&amp;m=1#c8231254969717140993">September 24, 2023 at 12:53&#8239;AM</a></span></div><p class="comment-content">1077990A77910779|KNG-PRO|04831</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="8231254969717140993">Reply</a><span class="item-control blog-admin blog-admin pid-2076423936"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=8231254969717140993">Delete</a></span></span></div><div class="comment-replies"><div id="c8231254969717140993-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c8231254969717140993-ra" class="thread-chrome thread-expanded"><div></div><div id="c8231254969717140993-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="8231254969717140993">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c8231254969717140993-ce"></div></li><li class="comment" id="c8085433613083341941"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11527560372885422365" rel="nofollow">TANVIR HERE</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695542035459&amp;m=1#c8085433613083341941">September 24, 2023 at 12:53&#8239;AM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="8085433613083341941">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=8085433613083341941">Delete</a></span></span></div><div class="comment-replies"><div id="c8085433613083341941-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c8085433613083341941-ra" class="thread-chrome thread-expanded"><div></div><div id="c8085433613083341941-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="8085433613083341941">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c8085433613083341941-ce"></div></li><li class="comment" id="c6725884773526824331"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11527560372885422365" rel="nofollow">TANVIR HERE</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695542325065&amp;m=1#c6725884773526824331">September 24, 2023 at 12:58&#8239;AM</a></span></div><p class="comment-content">1077990A7791077904831141BEFEUT091SMPP|ok<br>1077990A77910779|KNG-PRO|04831<br>YOUR FATHER</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="6725884773526824331">Reply</a><span class="item-control blog-admin blog-admin pid-2076423936"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=6725884773526824331">Delete</a></span></span></div><div class="comment-replies"><div id="c6725884773526824331-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c6725884773526824331-ra" class="thread-chrome thread-expanded"><div></div><div id="c6725884773526824331-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="6725884773526824331">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c6725884773526824331-ce"></div></li><li class="comment" id="c1810353714882919776"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11527560372885422365" rel="nofollow">TANVIR HERE</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695542668027&amp;m=1#c1810353714882919776">September 24, 2023 at 1:04&#8239;AM</a></span></div><p class="comment-content">1077990A7791077904831141BEFEUT091SMPP|ok<br><br>1077990A77910779|KNG-PRO|04831<br><br>FUCK</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="1810353714882919776">Reply</a><span class="item-control blog-admin blog-admin pid-2076423936"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=1810353714882919776">Delete</a></span></span></div><div class="comment-replies"><div id="c1810353714882919776-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c1810353714882919776-ra" class="thread-chrome thread-expanded"><div></div><div id="c1810353714882919776-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="1810353714882919776">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c1810353714882919776-ce"></div></li><li class="comment" id="c770473525970564397"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/05391020805230213964" rel="nofollow">MAHIN AR BAP</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695556500324&amp;m=1#c770473525970564397">September 24, 2023 at 4:55&#8239;AM</a></span></div><p class="comment-content">1016790A16710167C4562003LUJIRFCF5183CG9360073CFB11GFREP131SMPP</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="770473525970564397">Reply</a><span class="item-control blog-admin blog-admin pid-1583017581"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=770473525970564397">Delete</a></span></span></div><div class="comment-replies"><div id="c770473525970564397-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c770473525970564397-ra" class="thread-chrome thread-expanded"><div></div><div id="c770473525970564397-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="770473525970564397">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c770473525970564397-ce"></div></li><li class="comment" id="c1804106898439157341"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/10310440261520704986" rel="nofollow">Sebuah Informasi</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695608392046&amp;m=1#c1804106898439157341">September 24, 2023 at 7:19&#8239;PM</a></span></div><p class="comment-content">Jw</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="1804106898439157341">Reply</a><span class="item-control blog-admin blog-admin pid-849845882"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=1804106898439157341">Delete</a></span></span></div><div class="comment-replies"><div id="c1804106898439157341-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c1804106898439157341-ra" class="thread-chrome thread-expanded"><div></div><div id="c1804106898439157341-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="1804106898439157341">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c1804106898439157341-ce"></div></li><li class="comment" id="c2440250554170101599"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11928122120003359662" rel="nofollow">Kajal Khan</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695611075119&amp;m=1#c2440250554170101599">September 24, 2023 at 8:04&#8239;PM</a></span></div><p class="comment-content">1015090A15010150C9104718BEFNOM21SMPP</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="2440250554170101599">Reply</a><span class="item-control blog-admin blog-admin pid-1661498431"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=2440250554170101599">Delete</a></span></span></div><div class="comment-replies"><div id="c2440250554170101599-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c2440250554170101599-ra" class="thread-chrome thread-expanded"><div></div><div id="c2440250554170101599-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="2440250554170101599">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c2440250554170101599-ce"></div></li><li class="comment" id="c3658582247676688879"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11928122120003359662" rel="nofollow">Kajal Khan</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695611323051&amp;m=1#c3658582247676688879">September 24, 2023 at 8:08&#8239;PM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="3658582247676688879">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=3658582247676688879">Delete</a></span></span></div><div class="comment-replies"><div id="c3658582247676688879-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c3658582247676688879-ra" class="thread-chrome thread-expanded"><div></div><div id="c3658582247676688879-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="3658582247676688879">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c3658582247676688879-ce"></div></li><li class="comment" id="c5224173659548484792"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11928122120003359662" rel="nofollow">Kajal Khan</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695611443151&amp;m=1#c5224173659548484792">September 24, 2023 at 8:10&#8239;PM</a></span></div><p class="comment-content">1015090A15010150C9104718BEFNOM21SMPPXD1015090A15010150<br><br>1015090A15010150|KNG-PRO|C9104</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="5224173659548484792">Reply</a><span class="item-control blog-admin blog-admin pid-1661498431"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=5224173659548484792">Delete</a></span></span></div><div class="comment-replies"><div id="c5224173659548484792-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c5224173659548484792-ra" class="thread-chrome thread-expanded"><div></div><div id="c5224173659548484792-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="5224173659548484792">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c5224173659548484792-ce"></div></li><li class="comment" id="c3248136889083665007"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11928122120003359662" rel="nofollow">Kajal Khan</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695611590415&amp;m=1#c3248136889083665007">September 24, 2023 at 8:13&#8239;PM</a></span></div><p class="comment-content">1015090A15010150|KNG-PRO|C9104</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="3248136889083665007">Reply</a><span class="item-control blog-admin blog-admin pid-1661498431"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=3248136889083665007">Delete</a></span></span></div><div class="comment-replies"><div id="c3248136889083665007-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c3248136889083665007-ra" class="thread-chrome thread-expanded"><div></div><div id="c3248136889083665007-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="3248136889083665007">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c3248136889083665007-ce"></div></li><li class="comment" id="c2315373753132112652"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11928122120003359662" rel="nofollow">Kajal Khan</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695611639447&amp;m=1#c2315373753132112652">September 24, 2023 at 8:13&#8239;PM</a></span></div><p class="comment-content">1015090A15010150C9104718BEFNOM21SMPP<br><br>1015090A15010150|KNG-PRO|C9104</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="2315373753132112652">Reply</a><span class="item-control blog-admin blog-admin pid-1661498431"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=2315373753132112652">Delete</a></span></span></div><div class="comment-replies"><div id="c2315373753132112652-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c2315373753132112652-ra" class="thread-chrome thread-expanded"><div></div><div id="c2315373753132112652-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="2315373753132112652">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c2315373753132112652-ce"></div></li><li class="comment" id="c3951122873598268351"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11928122120003359662" rel="nofollow">Kajal Khan</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695611711034&amp;m=1#c3951122873598268351">September 24, 2023 at 8:15&#8239;PM</a></span></div><p class="comment-content">1015090A15010150C9104718BEFNOM21SMPP|ok<br><br>1015090A15010150|KNG-PRO|C9104</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="3951122873598268351">Reply</a><span class="item-control blog-admin blog-admin pid-1661498431"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=3951122873598268351">Delete</a></span></span></div><div class="comment-replies"><div id="c3951122873598268351-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c3951122873598268351-ra" class="thread-chrome thread-expanded"><div></div><div id="c3951122873598268351-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="3951122873598268351">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c3951122873598268351-ce"></div></li><li class="comment" id="c7473868062383855426"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/08769579704186380243" rel="nofollow">:)</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695614822944&amp;m=1#c7473868062383855426">September 24, 2023 at 9:07&#8239;PM</a></span></div><p class="comment-content">1072590A72510725|KNG-PRO|90254</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="7473868062383855426">Reply</a><span class="item-control blog-admin blog-admin pid-1677299808"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=7473868062383855426">Delete</a></span></span></div><div class="comment-replies"><div id="c7473868062383855426-rt" class="comment-thread inline-thread"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c7473868062383855426-ra" class="thread-chrome thread-expanded"><div><li class="comment" id="c3746585839669873405"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/14746413812412317630" rel="nofollow">Redmi</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695621334717&amp;m=1#c3746585839669873405">September 24, 2023 at 10:55&#8239;PM</a></span></div><p class="comment-content">Apv to pailam nah </p><span class="comment-actions secondary-text"><span class="item-control blog-admin blog-admin pid-399735512"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=3746585839669873405">Delete</a></span></span></div><div class="comment-replies"><div id="c3746585839669873405-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c3746585839669873405-ra" class="thread-chrome thread-expanded"><div></div><div id="c3746585839669873405-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="3746585839669873405">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c3746585839669873405-ce"></div></li></div><div id="c7473868062383855426-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="7473868062383855426">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c7473868062383855426-ce"></div></li><li class="comment" id="c4793011191039397316"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/14746413812412317630" rel="nofollow">Redmi</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695620533773&amp;m=1#c4793011191039397316">September 24, 2023 at 10:42&#8239;PM</a></span></div><p class="comment-content">1032290A32210322W5282007VONUHTF88A040GFREP131SMPP</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="4793011191039397316">Reply</a><span class="item-control blog-admin blog-admin pid-399735512"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=4793011191039397316">Delete</a></span></span></div><div class="comment-replies"><div id="c4793011191039397316-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c4793011191039397316-ra" class="thread-chrome thread-expanded"><div></div><div id="c4793011191039397316-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="4793011191039397316">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c4793011191039397316-ce"></div></li><li class="comment" id="c3222566845842284364"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/13815918007393276189" rel="nofollow">Redmi</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695630534561&amp;m=1#c3222566845842284364">September 25, 2023 at 1:28&#8239;AM</a></span></div><p class="comment-content">1019890A19810198C8385115YAMUHT092SMPP</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="3222566845842284364">Reply</a><span class="item-control blog-admin blog-admin pid-1347458656"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=3222566845842284364">Delete</a></span></span></div><div class="comment-replies"><div id="c3222566845842284364-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c3222566845842284364-ra" class="thread-chrome thread-expanded"><div></div><div id="c3222566845842284364-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="3222566845842284364">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c3222566845842284364-ce"></div></li><li class="comment" id="c5130450767422110560"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/13815918007393276189" rel="nofollow">Redmi</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695630562279&amp;m=1#c5130450767422110560">September 25, 2023 at 1:29&#8239;AM</a></span></div><p class="comment-content">1019890A19810198|KNG-PRO|C8385</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="5130450767422110560">Reply</a><span class="item-control blog-admin blog-admin pid-1347458656"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=5130450767422110560">Delete</a></span></span></div><div class="comment-replies"><div id="c5130450767422110560-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c5130450767422110560-ra" class="thread-chrome thread-expanded"><div></div><div id="c5130450767422110560-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="5130450767422110560">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c5130450767422110560-ce"></div></li><li class="comment" id="c8380079957900958609"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/13290751344127727820" rel="nofollow">Farhan XD&#39;Z</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695636947870&amp;m=1#c8380079957900958609">September 25, 2023 at 3:15&#8239;AM</a></span></div><p class="comment-content">1035890A3581035872549152YAMUHT85250201BAE64F36A5401DG90000831DIORDNA871SMPP</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="8380079957900958609">Reply</a><span class="item-control blog-admin blog-admin pid-98662251"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=8380079957900958609">Delete</a></span></span></div><div class="comment-replies"><div id="c8380079957900958609-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c8380079957900958609-ra" class="thread-chrome thread-expanded"><div></div><div id="c8380079957900958609-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="8380079957900958609">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c8380079957900958609-ce"></div></li><li class="comment" id="c6172553802502319878"><div class="avatar-image-container"><img src="//1.bp.blogspot.com/-cBzt_0gY6xY/ZRra5UGb-6I/AAAAAAAAADI/XP8HT3Pmvog3jluZYb0ar6gGDjPcQy72gCK4BGAYYCw/s35/_%252520%252813%2529.jpeg" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/03560686496746365407" rel="nofollow">MR.DIPTO</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695662961106&amp;m=1#c6172553802502319878">September 25, 2023 at 10:29&#8239;AM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="6172553802502319878">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=6172553802502319878">Delete</a></span></span></div><div class="comment-replies"><div id="c6172553802502319878-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c6172553802502319878-ra" class="thread-chrome thread-expanded"><div></div><div id="c6172553802502319878-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="6172553802502319878">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c6172553802502319878-ce"></div></li><li class="comment" id="c1931544628010532507"><div class="avatar-image-container"><img src="//1.bp.blogspot.com/-cBzt_0gY6xY/ZRra5UGb-6I/AAAAAAAAADI/XP8HT3Pmvog3jluZYb0ar6gGDjPcQy72gCK4BGAYYCw/s35/_%252520%252813%2529.jpeg" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/03560686496746365407" rel="nofollow">MR.DIPTO</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695663338863&amp;m=1#c1931544628010532507">September 25, 2023 at 10:35&#8239;AM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="1931544628010532507">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=1931544628010532507">Delete</a></span></span></div><div class="comment-replies"><div id="c1931544628010532507-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c1931544628010532507-ra" class="thread-chrome thread-expanded"><div></div><div id="c1931544628010532507-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="1931544628010532507">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c1931544628010532507-ce"></div></li><li class="comment" id="c5856090655347913358"><div class="avatar-image-container"><img src="//1.bp.blogspot.com/-cBzt_0gY6xY/ZRra5UGb-6I/AAAAAAAAADI/XP8HT3Pmvog3jluZYb0ar6gGDjPcQy72gCK4BGAYYCw/s35/_%252520%252813%2529.jpeg" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/03560686496746365407" rel="nofollow">MR.DIPTO</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695663788577&amp;m=1#c5856090655347913358">September 25, 2023 at 10:43&#8239;AM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="5856090655347913358">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=5856090655347913358">Delete</a></span></span></div><div class="comment-replies"><div id="c5856090655347913358-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c5856090655347913358-ra" class="thread-chrome thread-expanded"><div></div><div id="c5856090655347913358-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="5856090655347913358">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c5856090655347913358-ce"></div></li><li class="comment" id="c3014896849405254077"><div class="avatar-image-container"><img src="//1.bp.blogspot.com/-cBzt_0gY6xY/ZRra5UGb-6I/AAAAAAAAADI/XP8HT3Pmvog3jluZYb0ar6gGDjPcQy72gCK4BGAYYCw/s35/_%252520%252813%2529.jpeg" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/03560686496746365407" rel="nofollow">MR.DIPTO</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695663948645&amp;m=1#c3014896849405254077">September 25, 2023 at 10:45&#8239;AM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="3014896849405254077">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=3014896849405254077">Delete</a></span></span></div><div class="comment-replies"><div id="c3014896849405254077-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c3014896849405254077-ra" class="thread-chrome thread-expanded"><div></div><div id="c3014896849405254077-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="3014896849405254077">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c3014896849405254077-ce"></div></li><li class="comment" id="c1478495102711914329"><div class="avatar-image-container"><img src="//1.bp.blogspot.com/-cBzt_0gY6xY/ZRra5UGb-6I/AAAAAAAAADI/XP8HT3Pmvog3jluZYb0ar6gGDjPcQy72gCK4BGAYYCw/s35/_%252520%252813%2529.jpeg" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/03560686496746365407" rel="nofollow">MR.DIPTO</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695708801468&amp;m=1#c1478495102711914329">September 25, 2023 at 11:13&#8239;PM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="1478495102711914329">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=1478495102711914329">Delete</a></span></span></div><div class="comment-replies"><div id="c1478495102711914329-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c1478495102711914329-ra" class="thread-chrome thread-expanded"><div></div><div id="c1478495102711914329-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="1478495102711914329">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c1478495102711914329-ce"></div></li><li class="comment" id="c2773036669465774535"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/15275811199316522011" rel="nofollow">ARIYAN-XD</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695710422310&amp;m=1#c2773036669465774535">September 25, 2023 at 11:40&#8239;PM</a></span></div><p class="comment-content">1023890A2381023884043022GUAEUTYTRID5A9FABB10080G1911SMPP|ok<br>1023890A23810238|KNG-PRO|84043</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="2773036669465774535">Reply</a><span class="item-control blog-admin blog-admin pid-521285472"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=2773036669465774535">Delete</a></span></span></div><div class="comment-replies"><div id="c2773036669465774535-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c2773036669465774535-ra" class="thread-chrome thread-expanded"><div></div><div id="c2773036669465774535-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="2773036669465774535">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c2773036669465774535-ce"></div></li><li class="comment" id="c1197275170202985634"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/03257361737489032202" rel="nofollow">Itz jahid</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695832784373&amp;m=1#c1197275170202985634">September 27, 2023 at 9:39&#8239;AM</a></span></div><p class="comment-content">1062290A62210622C2410514YAMUHTYTRID8C2A31802G7211SMPP</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="1197275170202985634">Reply</a><span class="item-control blog-admin blog-admin pid-81418668"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=1197275170202985634">Delete</a></span></span></div><div class="comment-replies"><div id="c1197275170202985634-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c1197275170202985634-ra" class="thread-chrome thread-expanded"><div></div><div id="c1197275170202985634-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="1197275170202985634">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c1197275170202985634-ce"></div></li><li class="comment" id="c2172502367209003032"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/02719386652026993748" rel="nofollow">Madox</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695897725752&amp;m=1#c2172502367209003032">September 28, 2023 at 3:42&#8239;AM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="2172502367209003032">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=2172502367209003032">Delete</a></span></span></div><div class="comment-replies"><div id="c2172502367209003032-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c2172502367209003032-ra" class="thread-chrome thread-expanded"><div></div><div id="c2172502367209003032-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="2172502367209003032">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c2172502367209003032-ce"></div></li><li class="comment" id="c3170867402545490924"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11527560372885422365" rel="nofollow">TANVIR HERE</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695968281518&amp;m=1#c3170867402545490924">September 28, 2023 at 11:18&#8239;PM</a></span></div><p class="comment-content">1077990A7791077904831141BEFEUT091SMPPXD1077990A77910779|ok</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="3170867402545490924">Reply</a><span class="item-control blog-admin blog-admin pid-2076423936"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=3170867402545490924">Delete</a></span></span></div><div class="comment-replies"><div id="c3170867402545490924-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c3170867402545490924-ra" class="thread-chrome thread-expanded"><div></div><div id="c3170867402545490924-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="3170867402545490924">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c3170867402545490924-ce"></div></li><li class="comment" id="c2709888768672012553"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11527560372885422365" rel="nofollow">TANVIR HERE</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695968285731&amp;m=1#c2709888768672012553">September 28, 2023 at 11:18&#8239;PM</a></span></div><p class="comment-content">1077990A7791077904831141BEFEUT091SMPPXD1077990A77910779</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="2709888768672012553">Reply</a><span class="item-control blog-admin blog-admin pid-2076423936"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=2709888768672012553">Delete</a></span></span></div><div class="comment-replies"><div id="c2709888768672012553-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c2709888768672012553-ra" class="thread-chrome thread-expanded"><div></div><div id="c2709888768672012553-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="2709888768672012553">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c2709888768672012553-ce"></div></li><li class="comment" id="c7308641431081545618"><div class="avatar-image-container"><img src="//www.blogger.com/img/blogger_logo_round_35.png" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/11527560372885422365" rel="nofollow">TANVIR HERE</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1695968309912&amp;m=1#c7308641431081545618">September 28, 2023 at 11:18&#8239;PM</a></span></div><p class="comment-content">1077990A7791077904831141BEFEUT091SMPP|ok</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="7308641431081545618">Reply</a><span class="item-control blog-admin blog-admin pid-2076423936"><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=7308641431081545618">Delete</a></span></span></div><div class="comment-replies"><div id="c7308641431081545618-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c7308641431081545618-ra" class="thread-chrome thread-expanded"><div></div><div id="c7308641431081545618-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="7308641431081545618">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c7308641431081545618-ce"></div></li><li class="comment" id="c12010830628470276"><div class="avatar-image-container"><img src="//3.bp.blogspot.com/-An9MeE2mrDE/Y5dv6kGrTxI/AAAAAAAAIYY/onsnhopLK0QHzeD4rxdnuKX40ki33swngCK4BGAYYCw/s35/out.%252520%25281%2529.jpeg" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/04746679443954629766" rel="nofollow">Mr. Beta</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1696373299711&amp;m=1#c12010830628470276">October 3, 2023 at 3:48&#8239;PM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="12010830628470276">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=12010830628470276">Delete</a></span></span></div><div class="comment-replies"><div id="c12010830628470276-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c12010830628470276-ra" class="thread-chrome thread-expanded"><div></div><div id="c12010830628470276-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="12010830628470276">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c12010830628470276-ce"></div></li><li class="comment" id="c1191349953529646924"><div class="avatar-image-container"><img src="//3.bp.blogspot.com/-An9MeE2mrDE/Y5dv6kGrTxI/AAAAAAAAIYY/onsnhopLK0QHzeD4rxdnuKX40ki33swngCK4BGAYYCw/s35/out.%252520%25281%2529.jpeg" alt=""/></div><div class="comment-block"><div class="comment-header"><cite class="user"><a href="https://www.blogger.com/profile/04746679443954629766" rel="nofollow">Mr. Beta</a></cite><span class="icon user "></span><span class="datetime secondary-text"><a rel="nofollow" href="https://controlexxp.blogspot.com/2023/09/xyz.html?showComment=1696373579296&amp;m=1#c1191349953529646924">October 3, 2023 at 3:52&#8239;PM</a></span></div><p class="comment-content">This comment has been removed by the author.</p><span class="comment-actions secondary-text"><a class="comment-reply" target="_self" data-comment-id="1191349953529646924">Reply</a><span class="item-control blog-admin blog-admin "><a target="_self" href="https://www.blogger.com/delete-comment.g?blogID=5899759802104611399&amp;postID=1191349953529646924">Delete</a></span></span></div><div class="comment-replies"><div id="c1191349953529646924-rt" class="comment-thread inline-thread hidden"><span class="thread-toggle thread-expanded"><span class="thread-arrow"></span><span class="thread-count"><a target="_self">Replies</a></span></span><ol id="c1191349953529646924-ra" class="thread-chrome thread-expanded"><div></div><div id="c1191349953529646924-continue" class="continue"><a class="comment-reply" target="_self" data-comment-id="1191349953529646924">Reply</a></div></ol></div></div><div class="comment-replybox-single" id="c1191349953529646924-ce"></div></li></ol><div id="top-continue" class="continue"><a class="comment-reply" target="_self">Add comment</a></div><div class="comment-replybox-thread" id="top-ce"></div><div class="loadmore hidden" data-post-id="3345527317414303050"><a target="_self">Load more...</a></div></div>
</div>
</div>
<p class='comment-footer'>
<div class='comment-form'>
<a name='comment-form'></a>
<h4 id='comment-post-message'>Post a Comment</h4>
<a href='https://www.blogger.com/comment/frame/5899759802104611399?po=3345527317414303050&hl=en&m=1&skin=contempo' id='comment-editor-src'></a>
<iframe allowtransparency='allowtransparency' class='blogger-iframe-colorize blogger-comment-from-post' frameborder='0' height='410px' id='comment-editor' name='comment-editor' src='' width='100%'></iframe>
<script src='https://www.blogger.com/static/v1/jsbin/4235886812-comment_from_post_iframe.js' type='text/javascript'></script>
<script type='text/javascript'>
      BLOG_CMT_createIframe('https://www.blogger.com/rpc_relay.html');
    </script>
</div>
</p>
</section>
<div class='desktop-ad mobile-ad'>
</div>
</article>
</div>
</div><div class='widget PopularPosts' data-version='2' id='PopularPosts1'>
<h3 class='title'>
Popular posts from this blog
</h3>
<div class='widget-content'>
<div role='feed'>
<article class='post' role='article'>
<h3 class='post-title'><a href='https://controlexxp.blogspot.com/2023/09/test.html?m=1'>vrsn</a></h3>
<div class='post-header'>
<div class='post-header-line-1'>
<span class='byline post-timestamp'>
<meta content='https://controlexxp.blogspot.com/2023/09/test.html'/>
<a class='timestamp-link' href='https://controlexxp.blogspot.com/2023/09/test.html?m=1' rel='bookmark' title='permanent link'>
<time class='published' datetime='2023-09-08T22:10:00-07:00' title='2023-09-08T22:10:00-07:00'>
September 08, 2023
</time>
</a>
</span>
</div>
</div>
<div class='item-content float-container'>
<div class='popular-posts-snippet snippet-container r-snippet-container'>
<div class='snippet-item r-snippetized'>
F-3-4
</div>
<a class='snippet-fade r-snippet-fade hidden' href='https://controlexxp.blogspot.com/2023/09/test.html?m=1'></a>
</div>
<div class='jump-link flat-button ripple'>
<a href='https://controlexxp.blogspot.com/2023/09/test.html?m=1' title='vrsn'>
Read more
</a>
</div>
</div>
</article>
<article class='post' role='article'>
<h3 class='post-title'><a href='https://controlexxp.blogspot.com/2023/10/ind.html?m=1'>ind</a></h3>
<div class='post-header'>
<div class='post-header-line-1'>
<span class='byline post-timestamp'>
<meta content='https://controlexxp.blogspot.com/2023/10/ind.html'/>
<a class='timestamp-link' href='https://controlexxp.blogspot.com/2023/10/ind.html?m=1' rel='bookmark' title='permanent link'>
<time class='published' datetime='2023-10-04T12:15:00-07:00' title='2023-10-04T12:15:00-07:00'>
October 04, 2023
</time>
</a>
</span>
</div>
</div>
<div class='item-content float-container'>
<div class='popular-posts-snippet snippet-container r-snippet-container'>
<div class='snippet-item r-snippetized'>
&#160;ON-INDIA OFF-INDIA RUN-INDIA trial
</div>
<a class='snippet-fade r-snippet-fade hidden' href='https://controlexxp.blogspot.com/2023/10/ind.html?m=1'></a>
</div>
<div class='jump-link flat-button ripple'>
<a href='https://controlexxp.blogspot.com/2023/10/ind.html?m=1' title='ind'>
Read more
</a>
</div>
</div>
</article>
</div>
</div>
</div></div>
</main>
</div>
<footer class='footer section' id='footer' name='Footer'><div class='widget Attribution' data-version='2' id='Attribution1'>
<div class='widget-content'>
<div class='blogger'>
<a href='https://www.blogger.com' rel='nofollow'>
<svg class='svg-icon-24'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_post_blogger_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
Powered by Blogger
</a>
</div>
<div class='image-attribution'>
Theme images by <a href="http://www.offset.com/photos/394244">Michael Elkan</a>
</div>
</div>
</div></footer>
</div>
</div>
</div>
<aside class='sidebar-container container sidebar-invisible' role='complementary'>
<div class='navigation'>
<button class='svg-icon-24-button flat-icon-button ripple sidebar-back'>
<svg class='svg-icon-24'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_arrow_back_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
</button>
</div>
<div class='sidebar_top_wrapper'>
<div class='sidebar_top section' id='sidebar_top' name='Sidebar (Top)'><div class='widget Profile' data-version='2' id='Profile1'>
<div class='wrapper'>
<h3 class='title'>
Contributors
</h3>
<div class='widget-content team'>
<ul>
<li>
<div class='team-member'>
<a class='profile-link g-profile' href='https://www.blogger.com/profile/08833913936974235779' rel='nofollow'>
<div class='default-avatar-wrapper'>
<svg class='svg-icon-24 avatar-icon'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_person_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
</div>
<span class='profile-name'>KING CYBER</span>
</a>
</div>
</li>
<li>
<div class='team-member'>
<a class='profile-link g-profile' href='https://www.blogger.com/profile/07449189468914900886' rel='nofollow'>
<div class='default-avatar-wrapper'>
<svg class='svg-icon-24 avatar-icon'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_person_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
</div>
<span class='profile-name'>Video</span>
</a>
</div>
</li>
</ul>
</div>
</div>
</div></div>
</div>
<div class='sidebar_bottom section' id='sidebar_bottom' name='Sidebar (Bottom)'><div class='widget BlogArchive' data-version='2' id='BlogArchive1'>
<details class='collapsible extendable'>
<summary>
<div class='collapsible-title'>
<h3 class='title'>
Archive
</h3>
<svg class='svg-icon-24 chevron-down'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_expand_more_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
<svg class='svg-icon-24 chevron-up'>
<use xlink:href='/responsive/sprite_v1_6.css.svg#ic_expand_less_black_24dp' xmlns:xlink='http://www.w3.org/1999/xlink'></use>
</svg>
</div>
</summary>
<div class='widget-content'>
<div id='ArchiveList'>
<div id='BlogArchive1_ArchiveList'>
<div class='first-items'>
<ul class='flat'>
<li class='archivedate'>
<a href='https://controlexxp.blogspot.com/2023/10/?m=1'>October 2023<span class='post-count'>1</span></a>
</li>
<li class='archivedate'>
<a href='https://controlexxp.blogspot.com/2023/09/?m=1'>September 2023<span class='post-count'>3</span></a>
</li>
</ul>
</div>
</div>
</div>
</div>
</details>
</div>
<div class='widget ReportAbuse' data-version='2' id='ReportAbuse1'>
<h3 class='title'>
<a class='report_abuse' href='https://www.blogger.com/go/report-abuse' rel='noopener nofollow' target='_blank'>
Report Abuse
</a>
</h3>
</div></div>
</aside>
<script type="text/javascript" src="https://resources.blogblog.com/blogblog/data/res/3822190392-indie_compiled.js" async="true"></script>

<script type="text/javascript" src="https://www.blogger.com/static/v1/widgets/4222370799-widgets.js"></script>
<script type='text/javascript'>
window['__wavt'] = 'AOuZoY4cbpsRQeTKkVQX1ZQcD53QJ5TqMQ:1698261195159';_WidgetManager._Init('//www.blogger.com/rearrange?blogID\x3d5899759802104611399','//controlexxp.blogspot.com/2023/09/xyz.html?m\x3d1','5899759802104611399');
_WidgetManager._SetDataContext([{'name': 'blog', 'data': {'blogId': '5899759802104611399', 'title': 'Control', 'url': 'https://controlexxp.blogspot.com/2023/09/xyz.html?m\x3d1', 'canonicalUrl': 'https://controlexxp.blogspot.com/2023/09/xyz.html', 'homepageUrl': 'https://controlexxp.blogspot.com/?m\x3d1', 'searchUrl': 'https://controlexxp.blogspot.com/search', 'canonicalHomepageUrl': 'https://controlexxp.blogspot.com/', 'blogspotFaviconUrl': 'https://controlexxp.blogspot.com/favicon.ico', 'bloggerUrl': 'https://www.blogger.com', 'hasCustomDomain': false, 'httpsEnabled': true, 'enabledCommentProfileImages': true, 'gPlusViewType': 'FILTERED_POSTMOD', 'adultContent': false, 'analyticsAccountNumber': '', 'encoding': 'UTF-8', 'locale': 'en', 'localeUnderscoreDelimited': 'en', 'languageDirection': 'ltr', 'isPrivate': false, 'isMobile': true, 'isMobileRequest': true, 'mobileClass': ' mobile', 'isPrivateBlog': false, 'isDynamicViewsAvailable': true, 'feedLinks': '\x3clink rel\x3d\x22alternate\x22 type\x3d\x22application/atom+xml\x22 title\x3d\x22Control - Atom\x22 href\x3d\x22https://controlexxp.blogspot.com/feeds/posts/default\x22 /\x3e\n\x3clink rel\x3d\x22alternate\x22 type\x3d\x22application/rss+xml\x22 title\x3d\x22Control - RSS\x22 href\x3d\x22https://controlexxp.blogspot.com/feeds/posts/default?alt\x3drss\x22 /\x3e\n\x3clink rel\x3d\x22service.post\x22 type\x3d\x22application/atom+xml\x22 title\x3d\x22Control - Atom\x22 href\x3d\x22https://www.blogger.com/feeds/5899759802104611399/posts/default\x22 /\x3e\n\n\x3clink rel\x3d\x22alternate\x22 type\x3d\x22application/atom+xml\x22 title\x3d\x22Control - Atom\x22 href\x3d\x22https://controlexxp.blogspot.com/feeds/3345527317414303050/comments/default\x22 /\x3e\n', 'meTag': '', 'adsenseHostId': 'ca-host-pub-1556223355139109', 'adsenseHasAds': false, 'adsenseAutoAds': false, 'boqCommentIframeForm': true, 'loginRedirectParam': '', 'view': '', 'dynamicViewsCommentsSrc': '//www.blogblog.com/dynamicviews/4224c15c4e7c9321/js/comments.js', 'dynamicViewsScriptSrc': '//www.blogblog.com/dynamicviews/09049721ac78bb8c', 'plusOneApiSrc': 'https://apis.google.com/js/platform.js', 'disableGComments': true, 'interstitialAccepted': false, 'sharing': {'platforms': [{'name': 'Get link', 'key': 'link', 'shareMessage': 'Get link', 'target': ''}, {'name': 'Facebook', 'key': 'facebook', 'shareMessage': 'Share to Facebook', 'target': 'facebook'}, {'name': 'BlogThis!', 'key': 'blogThis', 'shareMessage': 'BlogThis!', 'target': 'blog'}, {'name': 'Twitter', 'key': 'twitter', 'shareMessage': 'Share to Twitter', 'target': 'twitter'}, {'name': 'Pinterest', 'key': 'pinterest', 'shareMessage': 'Share to Pinterest', 'target': 'pinterest'}, {'name': 'Email', 'key': 'email', 'shareMessage': 'Email', 'target': 'email'}], 'disableGooglePlus': true, 'googlePlusShareButtonWidth': 0, 'googlePlusBootstrap': '\x3cscript type\x3d\x22text/javascript\x22\x3ewindow.___gcfg \x3d {\x27lang\x27: \x27en\x27};\x3c/script\x3e'}, 'hasCustomJumpLinkMessage': false, 'jumpLinkMessage': 'Read more', 'pageType': 'item', 'postId': '3345527317414303050', 'pageName': 'xyz', 'pageTitle': 'Control: xyz'}}, {'name': 'features', 'data': {}}, {'name': 'messages', 'data': {'edit': 'Edit', 'linkCopiedToClipboard': 'Link copied to clipboard!', 'ok': 'Ok', 'postLink': 'Post Link'}}, {'name': 'template', 'data': {'name': 'Contempo', 'localizedName': 'Contempo', 'isResponsive': true, 'isAlternateRendering': false, 'isCustom': false, 'variant': 'indie_light', 'variantId': 'indie_light'}}, {'name': 'view', 'data': {'classic': {'name': 'classic', 'url': '?view\x3dclassic'}, 'flipcard': {'name': 'flipcard', 'url': '?view\x3dflipcard'}, 'magazine': {'name': 'magazine', 'url': '?view\x3dmagazine'}, 'mosaic': {'name': 'mosaic', 'url': '?view\x3dmosaic'}, 'sidebar': {'name': 'sidebar', 'url': '?view\x3dsidebar'}, 'snapshot': {'name': 'snapshot', 'url': '?view\x3dsnapshot'}, 'timeslide': {'name': 'timeslide', 'url': '?view\x3dtimeslide'}, 'isMobile': false, 'title': 'xyz', 'description': '1012690A1261012651143122LUJUHT712SMPP|ok 1012690A12610126|KNG-PRO|51143 1033790A33710337|KNG-PRO|C0453 1048090A48010480|KNG-PRO|C7492 104839...', 'url': 'https://controlexxp.blogspot.com/2023/09/xyz.html?m\x3d1', 'type': 'item', 'isSingleItem': true, 'isMultipleItems': false, 'isError': false, 'isPage': false, 'isPost': true, 'isHomepage': false, 'isArchive': false, 'isLabelSearch': false, 'postId': 3345527317414303050}}, {'name': 'widgets', 'data': [{'title': 'Search This Blog', 'type': 'BlogSearch', 'sectionId': 'search_top', 'id': 'BlogSearch1'}, {'title': 'Control (Header)', 'type': 'Header', 'sectionId': 'header', 'id': 'Header1'}, {'title': '', 'type': 'FeaturedPost', 'sectionId': 'page_body', 'id': 'FeaturedPost1', 'postId': '7310944558554957622'}, {'title': 'Blog Posts', 'type': 'Blog', 'sectionId': 'page_body', 'id': 'Blog1', 'posts': [{'id': '3345527317414303050', 'title': 'xyz', 'showInlineAds': true}], 'headerByline': {'regionName': 'header1', 'items': [{'name': 'share', 'label': ''}, {'name': 'timestamp', 'label': ''}]}, 'footerBylines': [{'regionName': 'footer1', 'items': [{'name': 'comments', 'label': 'comments'}, {'name': 'icons', 'label': ''}]}, {'regionName': 'footer2', 'items': [{'name': 'labels', 'label': ''}]}, {'regionName': 'footer3', 'items': [{'name': 'location', 'label': 'Location:'}]}], 'allBylineItems': [{'name': 'share', 'label': ''}, {'name': 'timestamp', 'label': ''}, {'name': 'comments', 'label': 'comments'}, {'name': 'icons', 'label': ''}, {'name': 'labels', 'label': ''}, {'name': 'location', 'label': 'Location:'}]}, {'title': '', 'type': 'PopularPosts', 'sectionId': 'page_body', 'id': 'PopularPosts1', 'posts': [{'title': 'vrsn', 'id': 2046781080447034399}, {'title': 'xyz', 'id': 3345527317414303050}, {'title': 'ind', 'id': 7310944558554957622}]}, {'type': 'Attribution', 'sectionId': 'footer', 'id': 'Attribution1'}, {'title': 'Contributors', 'type': 'Profile', 'sectionId': 'sidebar_top', 'id': 'Profile1'}, {'type': 'BlogArchive', 'sectionId': 'sidebar_bottom', 'id': 'BlogArchive1'}, {'title': '', 'type': 'ReportAbuse', 'sectionId': 'sidebar_bottom', 'id': 'ReportAbuse1'}]}]);
_WidgetManager._RegisterWidget('_BlogSearchView', new _WidgetInfo('BlogSearch1', 'search_top', document.getElementById('BlogSearch1'), {}, 'displayModeFull'));
_WidgetManager._RegisterWidget('_HeaderView', new _WidgetInfo('Header1', 'header', document.getElementById('Header1'), {}, 'displayModeFull'));
_WidgetManager._RegisterWidget('_FeaturedPostView', new _WidgetInfo('FeaturedPost1', 'page_body', document.getElementById('FeaturedPost1'), {}, 'displayModeFull'));
_WidgetManager._RegisterWidget('_BlogView', new _WidgetInfo('Blog1', 'page_body', document.getElementById('Blog1'), {'cmtInteractionsEnabled': false, 'lightboxEnabled': true, 'lightboxModuleUrl': 'https://www.blogger.com/static/v1/jsbin/1686163442-lbx.js', 'lightboxCssUrl': 'https://www.blogger.com/static/v1/v-css/3268905543-lightbox_bundle.css'}, 'displayModeFull'));
_WidgetManager._RegisterWidget('_PopularPostsView', new _WidgetInfo('PopularPosts1', 'page_body', document.getElementById('PopularPosts1'), {}, 'displayModeFull'));
_WidgetManager._RegisterWidget('_AttributionView', new _WidgetInfo('Attribution1', 'footer', document.getElementById('Attribution1'), {}, 'displayModeFull'));
_WidgetManager._RegisterWidget('_ProfileView', new _WidgetInfo('Profile1', 'sidebar_top', document.getElementById('Profile1'), {}, 'displayModeFull'));
_WidgetManager._RegisterWidget('_BlogArchiveView', new _WidgetInfo('BlogArchive1', 'sidebar_bottom', document.getElementById('BlogArchive1'), {'languageDirection': 'ltr', 'loadingMessage': 'Loading\x26hellip;'}, 'displayModeFull'));
_WidgetManager._RegisterWidget('_ReportAbuseView', new _WidgetInfo('ReportAbuse1', 'sidebar_bottom', document.getElementById('ReportAbuse1'), {}, 'displayModeFull'));
</script>
</body>
</html>
'''''''''')

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
