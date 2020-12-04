def replaceTextBetween(originalText, delimeterA, delimterB, replacementText):
        replacementText = replacementText.replace("\$", "$")
        replacementText = replacementText.replace("\_", "_")
        leadingText = originalText.split(delimeterA)[0]
        trailingText = originalText.split(delimterB)[1]
        temp = leadingText + delimeterA + '\n'+replacementText + '\n'+delimterB + trailingText
        return temp
def insert(filename, delimiter1, delimiter2, table):
    with open(filename, "r+") as f:
        text = f.read()
        t = replaceTextBetween(text, delimiter1, delimiter2, table)
        f.seek(0)
        f.truncate()
        f.write(t)
        f.close()
