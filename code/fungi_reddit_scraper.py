 
 # # # # # # # # # # # # # # # # # # # # # #              
 #         ('                              #            
 #         '|                              #            
 #         |'                              #            
 #        [::]                             #             
 #        [::]   _......_                  #                        
 #        [::].-'      _.-`.               #                           
 #        [:.'    .-. '-._.-`.             #                             
 #        [/ /\   |  \        `-..         #                                 
 #        / / |   `-.'      .-.   `-.      #                                    
 #       /  `-'            (   `.    `.    #                                      
 #      |           /\      `-._/      \   #                                       
 #      '    .'\   /  `.           _.-'|   #                                       
 #     /    /  /   \_.-'        _.':;:/    #                                      
 #   .'     \_/             _.-':;_.-'     #                                     
 #  /   .-.             _.-' \;.-'         #                                 
 # /   (   \       _..-'     |             #                             
 # \    `._/  _..-'    .--.  |             #                             
 #  `-.....-'/  _ _  .'    '.|             #                             
 #           | |_|_| |      | \  (o)       #                                   
 #      (o)  | |_|_| |      | | (\'/)      #                                    
 #     (\'/)/  ''''' |     o|  \;:;        #                                  
 #      :;  |        |      |  |/)         #                                 
 #  LGB  ;: `-.._    /__..--'\.' ;:        #                                  
 #           :;  `--' :;   :;              #                            
 #                                         # 
 # # # # # # # # # # # # # # # # # # # # # #
 #  ______                                 #
 #  |  ___|                                #
 #  | |_ _   _ _ __   __ _ _   _ ___       #
 #  |  _| | | | '_ \ / _` | | | / __|      #
 #  | | | |_| | | | | (_| | |_| \__ \      #
 #  \_|  \__,_|_| |_|\__, |\__,_|___/      #
 #   _____            __/ |                #
 #  /  ___|          |___/                 #
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
sub_list = ['ShroomID', 'fungus']
df = pd.DataFrame()
for sub in sub_list:
    last = ''
    try:
        for i in range(50):
            url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={sub}&is_reddit_media_domain=true&is_video=false&size=50&before={last}&is_crosspostable="true"'
            response = requests.get(url)
            x += 1
            print(f"Request {x}")
            result = response.json()
            with open(f'../json_reqs/Request_{x}.json', 'w') as f:
                json.dump(result, f)
            results_df = pd.DataFrame(result['data'])
            links = results_df['url']
            got_image = []
            for link in links:
                file_name = link.split('/')[-1]
                try:
                    urllib.request.urlretrieve(url = link, filename= f'../fungi_reddit_images/{file_name}')
                    got_image.append(True)
                except:
                    got_image.append(False)
            last = int(results_df['created_utc'].tail(1))
            se = pd.Series()
            results_df['got_image'] = got_image
            keep = ['subreddit', 'title', 'created_utc', 'score', 'num_comments','url', 'got_image']
            df = pd.concat([df, results_df[keep]])
    except:
        print("r/{sub} Error")
# %% fungi_reddit_images Got 
print(df[df['got_image']==True].shape[1])

# %% Filter out All but jpg and png
file_list=[]
for filename in os.listdir('../fungi_reddit_images'):
    if filename.split('.')[-1] != "jpg" and filename.split('.')[-1] != "png":
        os.remove(f'../fungi_reddit_images/{filename}')
    else:
        file_list.append(filename.split('.')[-1])


# %% fungi_reddit_images Kept
len(file_list)

# %%
