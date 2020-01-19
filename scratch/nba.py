import os
from bs4 import BeautifulSoup
import requests
import pandas as pd
from proxy_requests import ProxyRequests

url_root = "https://www.baseball-reference.com/"
url_root = "https://www.pro-football-reference.com/"
url_root = "https://www.basketball-reference.com/"

# Scrape Teams
# Scrape team names
# Scrape team schedule by year

# works, but can't use proxyRequests
# pd.read_html(url)
def all_team_names(url_root):
    url = os.path.join(url_root, "teams") + "/"
    r = ProxyRequests(url)
    r.get()
    # print ip used
    print(r.get_proxy_used())
    soup = BeautifulSoup(r.get_raw(), "html.parser")
    tabs = soup.find_all("table")
    # active franchise: tabs[0] bc two tables on url, then pd_read_html returns a list
    df_active = pd.read_html(tabs[0].prettify())[0]
    # filter to max years, which is the main franchise. Do you need this?

    # Extract all the hrefs for the active teams:
    team_a_links = tabs[0].find_all("a", href=True)
    team_names = {
        t["href"].replace("teams", "").replace("/", ""): t.text
        for t in team_a_links
        if "/teams/" in t["href"]
    }
    return team_names


team_names = all_team_names(url_root)

#


r2 = requests.get(url)
