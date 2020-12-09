import numpy as np
import os
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180

maxRows = 20
maxCols = 20
maps_dir = f"/home/aajay/meta-monsterkong/monsterkong_randomensemble/maps{maxRows}x{maxCols}_test/"
assets_dir = "/home/aajay/meta-monsterkong/monsterkong_randomensemble/assets/"
totalMaps = 1000

def genMap(maxRows, maxCols, mapIter, curriculumBool=False):
    mapTmp = np.zeros((maxRows, maxCols), dtype='int8')
    minSize = min(maxRows, maxCols)
    princessCol = np.random.randint(low=3, high=maxCols - 2)
    if curriculumBool:
        candidate= (maxRows - 2) * (1-(mapIter / totalMaps)) # Remember that row 0 is top row visually and maxRow is bottom row visually
        princessRow=  np.max([4, int(np.floor(candidate))])
        print("princessRow" + str(princessRow))
    else:
        princessRow = np.random.randint(low=3, high=maxRows-2)
    princessCell = np.asarray([princessRow, princessCol])
    currentLevel = princessCell[0]+1 #levels are y axis/row index of the matrix
    if princessCell[1] >= maxCols-3:
        platRight = princessCell[1]
    else:
        platRight = np.random.randint(low=princessCell[1], high=maxCols-3)
    if princessCell[1] <= 3:
        platLeft = 3
    else:
        platLeft = np.random.randint(low=3, high=princessCell[1])
    prevTransport = -99
    print("platLeft and right" + str(platLeft) + " "+ str(platRight))
    while currentLevel < maxRows-2:
        mapTmp[currentLevel, platLeft:platRight+1] = 1 #Draw platform based on left and right edges calculated in previous iteration
        if prevTransport == 0:
            transportType = np.random.choice(a=[0, 1], p=[.5, .5]) # 0 means jump, 1 means ladder to traverse from belowLevel to currentLevel
        elif prevTransport == 1 or prevTransport == -99:
            transportType = np.random.choice(a=[0, 1], p=[.7, .3]) # 0 means jump, 1 means ladder to traverse from belowLevel to currentLevel
        if transportType == 0: # if jumping, either the left or right edge of the current platform must be accessible
            if platLeft < 4 and platRight >maxCols - 5:
                transportType = 1
            else:
                if platLeft -1 >= 3 and platRight+1 <= maxCols-3:
                    jumpPt = np.random.choice(a=[platLeft-1, platRight+1])
                elif platLeft-1 > 2:
                    jumpPt = platLeft-1
                elif platRight+1 <= maxRows-3:
                    jumpPt = platRight+1
                else:
                    transportType == 1 #If no valid jump points, make a ladder
                print("Jump pt")
                print(jumpPt)
                if jumpPt >=  maxCols-3:
                    platRight =  maxCols-3
                else:
                    platRight = np.random.randint(low=jumpPt, high= maxCols-2)
                if 3 >= jumpPt:
                    platLeft = 3
                else:
                    platLeft = np.random.randint(low=3, high=jumpPt)   
                belowLevel = currentLevel + 3

        if transportType == 1: # if ladder, just bust the ladder up through any of the current plat's tiles
            if platLeft>=platRight+1:
                ladderPt = platLeft
            else:
                ladderPt = np.random.randint(low=platLeft, high=platRight+1)
            if ladderPt >=  maxCols-3:
                platRight = maxCols-4
            else:
                platRight = np.random.randint(low=ladderPt, high= maxCols-3)
            if 3 >= ladderPt:
                platLeft = 3
            else:
                platLeft = np.random.randint(low=3, high=ladderPt)
            if currentLevel+3 >= maxRows:
                belowLevel = maxRows-1
            else:
                belowLevel = np.random.randint(low=currentLevel+3, high=min(currentLevel+5, maxRows)) #the agent can only go from height X to a platform at height X+3    
            print(currentLevel-1)
            print(belowLevel+1)
            print(ladderPt)
            mapTmp[currentLevel-1:belowLevel+1,ladderPt] = 2
            print("Ladder pt")
            print(ladderPt)
#             if ladderPt < maxRows-4:
#                 print("ladder put on starting from " + str(currentLevel-1))
#                 mapTmp[currentLevel-1:belowLevel+1,ladderPt+1] = 2
#                 mapTmp[currentLevel-1:belowLevel+1,ladderPt+2] = 2
#             else:
#                 print("ladder put on starting from " + str(currentLevel-1))
#                 mapTmp[currentLevel-1:belowLevel+1,ladderPt-1] = 2
#                 mapTmp[currentLevel-1:belowLevel+1,ladderPt-2] = 2
        currentLevel = belowLevel
        prevTransport=transportType
    if currentLevel < maxRows - 1:
        mapTmp[currentLevel, platLeft:platRight+1] = 1
#     mapTmp[maxCols-3, 7:9] = 0
#     mapTmp[maxCols-2, 7:9] = 0
    mapTmp[princessCell[0], princessCell[1]] = 77
    wallCol = np.ones(maxRows)
    mapTmp[:,0] = wallCol
    mapTmp[:,maxCols-1] = wallCol
    wallRow = np.ones(maxCols)
    mapTmp[maxRows-1,:] = wallRow
    mapTmp[0,:] = wallRow
    
    mapTmp[maxRows-2][maxCols//2] = 0

    if mapTmp[maxRows-3][maxCols//2] == 2:
        mapTmp[maxRows-2][maxCols//2] = 2    
    
    return mapTmp, princessCell


def merge_lr(img1, img2):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    (width1, height1) = img1.size
    (width2, height2) = img2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=img1, box=(0, 0))
    result.paste(im=img2, box=(width1, 0))
    return result

def merge_tb(img1, img2):
    (width1, height1) = img1.size
    (width2, height2) = img2.size

    result_width = max(width1, width2)
    result_height = height1 + height2

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=img1, box=(0, 0))
    result.paste(im=img2, box=(0, height1))
    return result

def genMapImage(map_arr):
    boundary = Image.open(assets_dir + 'boundary.png')
    boundary.thumbnail((15, 15), Image.ANTIALIAS)
    # plt.figure()
    # imshow(np.asarray(boundary))
    # Image._show(boundary)
    ladder = Image.open(assets_dir + 'ladder copy.png')
    ladder.thumbnail((15, 15), Image.ANTIALIAS)
    # plt.figure()
    # imshow(np.asarray(ladder))
    wall = Image.open(assets_dir + 'wood_block copy.png')
    wall.thumbnail((15, 15), Image.ANTIALIAS)
    # plt.figure()

    # imshow(np.asarray(wall))
    princess = Image.open(assets_dir + 'princess copy.png')
    princess.thumbnail((15, 15), Image.ANTIALIAS)
    # plt.figure()

    # imshow(np.asarray(princess))
    tileDict = {9: boundary, 0: boundary, 1: wall, 77: princess, 2: ladder}

    res = tileDict[map_arr[0][0]]
    for idx in range(1, len(map_arr[0])):
        res = merge_lr(res, tileDict[map_arr[0][idx]]) # Make row 1
    # plt.figure()
    # plt.title("Row 0")
    # imshow(np.asarray(res))

#     print(len(map_[:,0]))
    for rowIdx in range(1, len(map_arr[:,0])):
        nextRow = tileDict[map_arr[rowIdx][0]]
        for colIdx in range(1, len(map_arr[0])):
            nextRow = merge_lr(nextRow, tileDict[map_arr[rowIdx][colIdx]])
    #     plt.figure()
    #     plt.title("Row " + str(rowIdx))
    #     imshow(np.asarray(nextRow))
    #     Image._show(nextRow)
        res = merge_tb(res, nextRow)
    return res

def loadMap(i):
    file = open ( maps_dir + 'map' + str(i) + '.txt' , 'r')
    loaded = [list(__builtins__.map(int,line.split(','))) for line in file if line.strip() != "" ] #load your own custom map here
    out = np.array(loaded, dtype='uint8')
    return out

def generateImages():
    for i in range(totalMaps):
        lm = loadMap(i)
        if not os.path.isdir(maps_dir + 'images/'):
            os.mkdir(maps_dir + 'images/')
        genMapImage(lm).save(maps_dir + 'images/' + str(i) + '.png')

def main():
    if not os.path.isdir(maps_dir):
        os.mkdir(maps_dir)

    for i in range(totalMaps):
        print("Map " + str(i))
        mrMap, mrPrincessCell = genMap(maxRows, maxCols, mapIter=i)
        np.savetxt(maps_dir + 'map' + str(i) + '.txt', mrMap, delimiter=',', fmt='%d')
        fn = maps_dir + 'princess' + str(i) + '.txt'
        np.savetxt(fn, mrPrincessCell * 15 + 15 / 2, delimiter=',', fmt='%d')

    generateImages()

if __name__ == '__main__':
    main()