import numpy as np

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def edgelength(p0,p1):
    return np.linalg.norm(np.array(p1)-np.array(p0))

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

############################################
# ### compute the equiangular skewness ### #  
############################################
def compute_skewness(ELEMENTS,POINTS):
  skewlist = []
  print("*** computing mesh angles")
  for e in ELEMENTS:
    # triangles, loop over the angles
    theta_max = -1000.0
    theta_min = +1000.0
    # loop over the points on the element
    for i in range(len(e)):
        ROT = e[i:] + e[:i]
        # the 3 points
        p0 = POINTS[ROT[0]]
        p1 = POINTS[ROT[1]]
        p2 = POINTS[ROT[2]]
        v1 = [p2[0]-p1[0],p2[1]-p1[1]]
        v2 = [p0[0]-p1[0],p0[1]-p1[1]]
        theta = angle_between(v1,v2)*180.0/3.14159265
        theta_max = max(theta,theta_max)
        theta_min = min(theta,theta_min)        
    
    if (len(e)==3):
        theta_e = 60.0
    else:      
        theta_e = 90.0
    skewness = max((theta_max-theta_e)/(180.0-theta_e),(theta_e-theta_min)/theta_e)

        
    skewlist.append(skewness)
  skewlist.sort()        
  return(skewlist) 

# ############################################ #
# get the index of a keyword in a list
# ############################################ #
def getIndex(keyword,the_list):
  list = []
  for i, s in enumerate(the_list):
    if keyword in s:
      list.append(i)
  return list
# ############################################ #

# ############################################ #
# get the value of the keyword in the list
# ############################################ #
def getValue(keyword,the_list):
  list=[]
  for i, s in enumerate(the_list):
    if keyword in s:
      # remove any whitespaces and split on the equal sign  
      list.append(int(s.strip().split('=')[1]))
  return list
# ############################################ #


#########################################################
# ### create the list of connectivities             ### #
#########################################################
def createSpiderList(ELEMENTS,POINTS):
  NELEM = len(ELEMENTS)
  NPOIN = len(POINTS)  
  # the list of the connecting points for all NPOIN points.
  # we split squares into triangles and take into account all corner nodes of the square.
  SPIDERLIST = [[] for _ in range(NPOIN)]

  # loop over connectivities
  for e in ELEMENTS:
    if (len(e)==4):
        for i in range(4):
            ROT = e[i:] + e[:i]
            SPIDERLIST[ROT[1]].extend([ROT[0],ROT[2]])
    else:
        for i in range(3):
            ROT = e[i:] + e[:i]
            SPIDERLIST[ROT[1]].extend([ROT[0],ROT[2]])
        
  for i in range(NPOIN):
    SPIDERLIST[i] = list(dict.fromkeys(SPIDERLIST[i]))
  return SPIDERLIST
    


def laplacian(SPIDERLIST, EXCLUDELIST, POINTS):
  NEWPOINTS=[]
  for e in range(len(SPIDERLIST)):
    if e in EXCLUDELIST:
      NEWPOINTS.append([POINTS[e][0],POINTS[e][1]])
    else:
      dx = 0
      dy = 0
      Nw = 0
      for c in SPIDERLIST[e]:
        p0 = POINTS[e]
        p1 = POINTS[c]
        w = 1.0/edgelength(p0,p1)     
        Nw += w
        dx += w*(p1[0] - p0[0])
        dy += w*(p1[1] - p0[1])
        
      newx = p0[0] + dx / Nw
      newy = p0[1] + dy / Nw

      NEWPOINTS.append([newx,newy])     
  return NEWPOINTS


def checkMeshQuality(filename, zoneID=0):

  print('   Checking mesh quality of ' + filename,flush=True)

  with open(filename) as f:
    lines = [line.rstrip().expandtabs(1) for line in f]
 
  NELEM = getValue('NELEM',lines)[zoneID]
  NPOIN = getValue('NPOIN',lines)[zoneID]
  NELEM_index = getIndex('NELEM',lines)[zoneID]
  NPOIN_index = getIndex('NPOIN',lines)[zoneID]
  NMARKER= getValue('MARKER_ELEMS',lines)
  NMARKER_index= getIndex('MARKER_ELEMS',lines)
  print("location of NELEM=",getIndex('NELEM',lines)[zoneID])
  print("value of NELEM=",getValue('NELEM',lines)[zoneID])
  print("location of NPOINT=",getIndex('NPOIN',lines)[zoneID])
  print("value of NPOINT=",getValue('NPOIN',lines)[zoneID])
  print("location of MARKERS=",NMARKER_index)
  print("value of MARKERS=",NMARKER)

  ELEMENTS = []
  for e in range(NELEM):
    connectivities = lines[NELEM_index+e+1].split(' ')
    connectivities = list(map(int,connectivities[1:-1]))
    ELEMENTS.append(connectivities)

  POINTS = []
  for p in range(NPOIN):
    coords = lines[NPOIN_index+p+1].split(' ')
    coords = list(map(float,coords[:-1]))
    POINTS.append(coords)

  SPIDERLIST = createSpiderList(ELEMENTS,POINTS)

  EXCLUDELIST=[]
  for i in range(len(NMARKER)):
    for j in range(NMARKER[i]):
        pts = list(map(int,lines[NMARKER_index[i]+j+1][1:].strip().split(' ')))
        EXCLUDELIST += pts

  EXCLUDELIST = list(dict.fromkeys(EXCLUDELIST))
  EXCLUDELIST.sort()

  skew = compute_skewness(ELEMENTS,POINTS)
  return skew

