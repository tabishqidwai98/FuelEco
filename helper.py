

def cleanChangeVal(val):
    if isinstance(val,str):
        return float(val.replace('%',''))
    else:
        return val