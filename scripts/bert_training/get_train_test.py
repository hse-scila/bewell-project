def normalize_target(data, who_score="DF_who_score", diener_score="DF_diener_score"):
    data[who_score] = (data[who_score] - 5) / 25.
    data[diener_score] = (data[diener_score] - 5) / 30.
    return data

