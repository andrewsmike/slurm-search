"""
Generate human-friendly random phrases and file names.
"""
__all__ = [
    'random_phrase',
    'random_file_name',
]

from os.path import exists, join
from random import choice

def random_phrase(short=False):
    if short:
        words = [
            choice(adjectives),
            choice(adjectives),
            choice(nouns),
        ]
    else:
        words = [
            choice(adjectives),
            choice(adjectives),
            choice(adjectives),
            choice(nouns),
        ]

    return "_".join(words)

def random_file_name(path):
    phrase = random_phrase()
    while exists(join(path, phrase)):
        phrase = random_phrase()
    return phrase

adjectives = ["lumpy","roomy","vague","abrupt","free","tawdry","ripe","greedy","ritzy","short","brown","first","loose","wacky","light","thin","severe","hard","curved","yellow","tense","nutty","husky","mean","acrid","bent","clumsy","past","ablaze","four","rotten","tangy","lovely","smart","taboo","woozy","empty","aware","next","gaudy","quack","homely","bad","trite","cloudy","wiggly","overt","crazy","jolly","jagged","smooth","spicy","better","tested","loud","bloody","lean","little","foamy","elated","bouncy","wrong","mature","nifty","plucky","burly","filthy","brawny","quick","decent","nosy","nine","long","real","placid","every","groovy","gray","ragged","trashy","ten","tame","sordid","wide","used","rapid","big","silly","purple","second","tough","gabby","untidy","stale","gaping","narrow","rural","entire","fresh","hushed"]

nouns = ["army", "beer", "hat", "law", "guitar", "orange", "loss", "pie", "volume", "music", "son", "world", "dinner", "drama", "heart", "unit", "two", "art", "blood", "night", "bonus", "nation", "dealer", "tooth", "data", "growth", "idea", "truth", "skill", "media", "owner", "ratio", "lake", "wood", "area", "region", "week", "moment", "police", "breath", "camera", "height", "cell", "fact", "youth", "power", "apple", "engine", "editor", "queen", "thanks", "month", "video", "method", "salad", "phone", "extent", "tongue", "virus", "poetry", "movie", "safety", "girl", "client", "cousin", "people", "county", "actor", "driver", "gate", "hair", "user", "piano", "debt", "exam", "mode", "hall", "desk", "map", "shirt", "hotel", "honey", "basket", "lab", "math", "recipe", "oven", "church", "speech", "song", "king", "role", "length", "office", "ad", "city", "injury", "person", "memory", "leader"]

verbs = ["crash","instruct","answer","invest","mutter","going","expose","travel","dig","explode","schedule","install","resign","release","identify","block","strain","opt","commission","assert","stop","file","own","understand","launch","amend","touch","await","execute","shut","communicate","formulate","incur","describe","drag","remind","conceal","creep","neglect","end","beg","gaze","encounter","fail","bid","debate","match","promise","read","notice","classify","operate","comprise","top","sponsor","prompt","accumulate","introduce","set","secure","guarantee","carve","consider","voice","earn","supplement","shout","perform","substitute","deliver","need","struggle","fix","accept","discover","part","manufacture","listen","emphasise","check","rescue","mix","help","concede","organise","reach","recover","feature","draw","erect","target","access","rely","fetch","modify","inform","keep","ease","edit","slow"]


if __name__ == "__main__":
    print(random_phrase())
