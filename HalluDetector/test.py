# from decomposition import decompose_workflow_to_cache

# if __name__ == '__main__':
#     decompose_workflow_to_cache('/data2/yuhaoz/DeepResearch/HalluBench/data/close-source/qwen/Psychological_Personality.json')

import aiohttp
import logging
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from typing import List, Dict


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DESKTOP_UA   = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)

async def another_candidate_fetch_pages_async(urls: List[str]) -> Dict[str, str]:
    """
    Asynchronously fetches web pages, parses them to make links absolute,
    and returns the text content.
    """
    results: Dict[str, str] = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=DESKTOP_UA,
            locale="en-US",
            viewport={"width": 1280, "height": 800}
        )
        page = await context.new_page()
        for url in urls:
            try:
                await page.goto(url, timeout=60000)
                html = await page.content()
                soup = BeautifulSoup(html, "lxml")

                # Find all hyperlinks and replace them with "text (absolute_url)"
                for a in soup.find_all("a", href=True):
                    text = a.get_text(strip=True) or "[No Text]"
                    raw_href = a["href"]
                    full_href = urljoin(url, raw_href)
                    a.replace_with(f"{text} ({full_href})")
                
                # Extract text from the modified body content
                if soup.body:
                    content_with_links = soup.body.get_text(separator="\n", strip=True)
                else:
                    content_with_links = soup.get_text(separator="\n", strip=True)
                
                results[url] = content_with_links

            except PlaywrightTimeoutError:
                results[url] = "[Error] Timeout"
            except Exception as e:
                results[url] = f"[Error] {e!r}"

        await context.close()
        await browser.close()
    return results


async def fetch_url_with_retry(session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> str:
    """
    Fetch a single URL with retry logic.
    
    Args:
        session: aiohttp session
        url: URL to fetch
        max_retries: Maximum number of retry attempts
        
    Returns:
        Content of the URL or error message
    """
    for attempt in range(max_retries):
        logger.info(f"Fetching URL {url} on attempt {attempt + 1}")
        try:
            jina_url = f"https://r.jina.ai/{url}"
            async with session.get(jina_url, timeout=aiohttp.ClientTimeout(total=200)) as response:
                if response.status == 200:
                    content = await response.text()
                    if "ERROR" not in content:
                        logger.info(f"✅ Successfully fetched {url} ({len(content)} chars) on attempt {attempt + 1}")
                        return content
                else:
                    logger.info(f"❌ Failed to fetch {url} on attempt {attempt + 1} with iniital method!")
                    try:
                        web_content = await another_candidate_fetch_pages_async([url])
                        return web_content[url]
                    except Exception as e:
                        logger.warning(f"[Error] Playwright fetch failed for {url}: {e}")
                        error_msg = f"[Error] HTTP {response.status} (Playwright fallback failed: {e})"
                        return error_msg
        except Exception as e:
            logger.warning(f"[Error] Initial fetch failed for {url}: {e}")
            error_msg = f"[Error] Initial fetch failed: {e}"
            return error_msg
                    
    
    return f"[Error] Failed after {max_retries} attempts"


urls = [
    "https://www.levels.fyi/companies/openai/jobs",
    "https://www.metacareers.com/blog/building-pathways-for-students-in-ar-and-vr",
    "https://www.metacareers.com/jobs/1024912282791680",
    "https://timesofindia.indiatimes.com/education/news/what-its-really-like-to-work-at-openai-no-emails-12-hour-days-and-the-push-for-perfection/articleshow/122620842.cms",
    "https://timesofindia.indiatimes.com/technology/tech-news/meta-hires-two-more-openai-top-researchers-amid-300-million-ai-talent-war-report/articleshow/122567069.cms",
    "https://www.ziprecruiter.com/Jobs/Deepmind-Google",
    "https://www.ziprecruiter.com/Jobs/Deepmind-Google/--in-California",
    "https://www.ziprecruiter.com/Jobs/Full-Time-Meta-Ai",
    "https://www.ziprecruiter.com/Jobs/Meta-Ai",
    "https://www.ziprecruiter.com/Jobs/Openai",
    "https://www.ziprecruiter.com/co/Openai/Jobs"
]
# Fetch all URLs
async def main():
    web_content = {}
    async with aiohttp.ClientSession() as session:
        for i, url in enumerate(urls, 1):
            logger.info(f"Fetching URL {i}/{len(urls)}: {url}")
            content = await fetch_url_with_retry(session, url)
            web_content[url] = content

if __name__ == "__main__":
    asyncio.run(main())