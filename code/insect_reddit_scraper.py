 
 # # # # # # # # # # # # # # # # # # # # # # 
 #              .--.       .--.            #       
 #         _  `    \     /    `  _         #          
 #          `\.===. \.^./ .===./`          #         
 #                 \/`"`\/                 #  
 #              ,  |     |  ,              #     
 #             / `\|`-.-'|/` \             #      
 #            /    |  \  |    \            #       
 #         .-' ,-'`|   ; |`'-, '-.         #          
 #             |   |    \|   |             #       
 #             |   |    ;|   |             #      
 #             |   \    //   |             #      
 #             |    `._//'   |             #      
 #            .'             `.            #       
 #         _,'                 `,_         #          
 #         `                     `         #          
 # # # # # # # # # # # # # # # # # # # # # #
 #   _____                    _            #
 # |_   _|                  | |            #
 #   | | _ __  ___  ___  ___| |_           #
 #   | || '_ \/ __|/ _ \/ __| __|          #
 #  _| || | | \__ \  __/ (__| |_           #
 #  \___/_| |_|___/\___|\___|\__|          #                 
 #  /  ___|                                #
 #  \ `--.  ___ _ __ __ _ _ __   ___ _ __  #
 #   `--. \/ __| '__/ _` | '_ \ / _ \ '__| #
 #  /\__/ / (__| | | (_| | |_) |  __/ |    #
 #  \____/ \___|_|  \__,_| .__/ \___|_|    #
 #                       | |               #
 #                       |_|               #
 # # # # # # # # # # # # # # # # # # # # # #
# %% Imports
import json
import requests
import pandas as pd
import urllib.request
import os
# %% Scraper
x = 0
sub_list = ['insects']
df = pd.DataFrame()
for sub in sub_list:
    last = ''
    for i in range(50):
        url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={sub}&is_reddit_media_domain=true&is_video=false&size=50&before={last}&is_crosspostable="true"'
        response = requests.get(url)
        x += 1
        result = response.json()
        print(f"Request {x}")
        results_df = pd.DataFrame(result['data'])
        links = results_df['url']
        got_image = []
        for link in links:
            try:
                file_name = link.split('/')[-1]
                urllib.request.urlretrieve(url = link, filename= f'../insect_reddit_images/{file_name}')
                got_image.append(True)
            except:
                print("No Image")
                got_image.append(False)
        last = int(results_df['created_utc'].tail(1))
        se = pd.Series()
        results_df['got_image'] = got_image
        keep = ['subreddit', 'title', 'created_utc', 'score', 'num_comments','url', 'got_image']
        df = pd.concat([df, results_df[keep]])
# %% Images Got

# %% Filter out All but jpg and png
file_list=[]
for filename in os.listdir('../insect_reddit_images'):
    if filename.split('.')[-1] != "jpg" and filename.split('.')[-1] != "png":
        os.remove(f'../insect_reddit_images/{filename}')
    else:
        file_list.append(filename.split('.')[-1])


# %% Images Kept
len(file_list)
# %%
