import numpy as np
lst = np.array([1900, 780, 2210, 920, 330, 1460, 440, 610, 1560, 1210, 3700, 470, 450, 240, 310, 250, 430,
       680, 380, 480, 1340, 1030, 1340, 1020, 550, 1050, 170, 500, 790])
beads = np.array(["83110", "17383", "10070", "17070", "10020", "53430", "89110", "38657", "57290", "53270",
                  "38681", "37188", "07332", "38395/1", "38398/1", "97050", "97070", "90090", "97120", "17110",
                  "17140", "01113","38318/1","46318", "17090", "10090",  "68080" , "33210", "67100" ])
QTY = 420

str = "#. <a href=\"http://www.perlinissima.it/prodotto/#/\" target=\"_blank\">#</a> ()"

print('{}. <a href="http://www.perlinissima.it/prodotto/{}/" target="_blank">#</a> ({})'.format(1,"sss", 2))
print(len(np.unique(beads)))
print("total number of colors", len(lst))

def packs(qty):
    rest = qty % QTY
    if rest == 0:
        return int(qty / QTY)
    else:
        return int(qty / QTY) + 1

packQty = list(map(packs,  lst))
for i, q, p, b in zip(range(1, len(lst) + 1), lst, packQty, beads):
    # print(i, q, p, b)
    print('{}. <a href="http://www.perlinissima.it/prodotto/{}/" target="_blank">{}</a> (n. {}, {} bustine)'.format(i, b, b, q, p))
print(sum(packQty))

print(8 + (sum(packQty))*0.5)