from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor


class OcbcSpider(scrapy.Spider):
    name = "ocbc_spider"
    allowed_domains = ["ocbc.com"]
    start_urls = [
        "https://www.ocbc.com/personal-banking/investments/precious-metals-account",
        "https://www.ocbc.com/personal-banking/security/secure-banking-ways/dispute-card-transactions",
        "https://www.ocbc.com/personal-banking/security/secure-banking-ways/ocbc-moneylock"
    ]

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        # stops at depth of 3
        "DEPTH_LIMIT": 3,
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 0.25,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.25,
        "AUTOTHROTTLE_MAX_DELAY": 3.0,
        "USER_AGENT": "ocbc-rag-crawler/0.1 (+https://www.ocbc.com/)",
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = Path(__file__).resolve().parent / "scraped_html_files"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Finds all links within scraped files and follows them for each, recusively
        self.link_extractor = LinkExtractor(
            allow_domains=self.allowed_domains,
            deny_extensions=set(),
            unique=True,
        )

    def parse(self, response: scrapy.http.Response):
        if self._is_html_response(response):
            self._store_response_html(response)

        for link in self.link_extractor.extract_links(response):
            yield response.follow(link.url, callback=self.parse)

    def _is_html_response(self, response: scrapy.http.Response) -> bool:
        content_type = response.headers.get(b"Content-Type", b"").decode("latin1")
        return "text/html" in content_type

    def _store_response_html(self, response: scrapy.http.Response) -> None:
        url = response.url
        parsed = urlparse(url)
        path = parsed.path.strip("/") or "index"
        safe_path = re.sub(r"[^A-Za-z0-9._-]+", "_", path)
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        filename = f"{parsed.netloc}_{safe_path}_{url_hash}.html"
        file_path = self.output_dir / filename

        file_path.write_bytes(response.body)


if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(OcbcSpider)
    process.start()
