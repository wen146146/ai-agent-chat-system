import requests
from bs4 import BeautifulSoup
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool, BaseTool


class WebSearchInput(BaseModel):
    """联网搜索工具输入参数模型"""
    query: str = Field(description="搜索关键词")
    num_results: int = Field(default=5, ge=1, le=10, description="返回结果数量，默认 5 条，最多 10 条")


class WebPageFetchInput(BaseModel):
    """网页抓取工具输入参数模型"""
    url: str = Field(description="要抓取的网页地址")
    max_chars: int = Field(default=3000, description="返回内容最大字符数，默认 3000")


@tool(args_schema=WebSearchInput)
def web_search(query: str, num_results: int = 5) -> str:
    """
    联网搜索工具，输入搜索关键词，返回相关网页的标题、摘要和链接。
    适用于需要实时信息、最新资讯、事实查询等场景。
    """
    num_results = min(num_results, 10)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select(".result")[:num_results]:
            title_el = item.select_one(".result__title")
            snippet_el = item.select_one(".result__snippet")
            link_el = item.select_one(".result__url")

            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            snippet = snippet_el.get_text(strip=True) if snippet_el else "无摘要"
            link = link_el.get("href", "").strip() if link_el else ""
            if link and link.startswith("//"):
                link = "https:" + link

            results.append({
                "title": title,
                "snippet": snippet,
                "url": link
            })

        if not results:
            return f'[结果] 搜索 "{query}" 未找到相关内容，请尝试更换关键词。'

        lines = [f'搜索关键词: "{query}" 共找到 {len(results)} 条结果:']
        for i, r in enumerate(results, 1):
            lines.append(f"\n--- 第 {i} 条 ---")
            lines.append(f"标题: {r['title']}")
            lines.append(f"摘要: {r['snippet']}")
            lines.append(f"链接: {r['url']}")

        return "\n".join(lines)

    except requests.RequestException as e:
        return f"[错误] 网络请求失败: {str(e)}"
    except Exception as e:
        return f"[错误] 搜索解析失败: {str(e)}"


@tool(args_schema=WebPageFetchInput)
def web_fetch(url: str, max_chars: int = 3000) -> str:
    """
    网页内容抓取工具，输入 URL 地址，抓取并返回网页的纯文本内容。
    适用于需要查看某个网页详细内容的场景，可配合 web_search 使用。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + f"\n\n... (内容过长，已截断至 {max_chars} 字符)"

        if not clean_text.strip():
            return f"[结果] 网页 {url} 未能提取到有效文本内容"

        return f"网页: {url}\n\n{clean_text}"

    except requests.RequestException as e:
        return f"[错误] 网页请求失败: {str(e)}"
    except Exception as e:
        return f"[错误] 内容提取失败: {str(e)}"


def get_all_search_tools() -> List[BaseTool]:
    """工厂函数：返回所有搜索相关工具实例"""
    return [web_search, web_fetch]
