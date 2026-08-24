import random,time,json
from collections import deque,defaultdict
random.seed(13)
NSCR=100;ST_OK,ST_DEG,ST_FAIL=0,1,2
RLS=[0.98,0.7,0.4,0.12];AGS=["agt_a","agt_b","agt_g","agt_d"];OPS=["op_k","op_l","op_c","op_p","op_s"]
def mkS(i):return{"id":f"s{i:03d}","rl":random.choice(RLS),"st":ST_OK,"fq":0,"ok":0}
S=[mkS(i)for i in range(NSCR)]
def tk(s):
 r=random.random()
 if r>s["rl"]:
  s["st"]=ST_FAIL if r>s["rl"]+0.15 else ST_DEG;s["fq"]+=1
 else:s["ok"]+=1;s["st"]=ST_OK
 return s["st"]
def ocr(n):
 d=deque(OPS);d.rotate(n%len(OPS));return d[0]
def dtI():
 o=[]
 for s in S:
  if tk(s)!=ST_OK:o.append({"sid":s["id"],"st":s["st"],"rl":s["rl"]})
 return o
def agI(i):
 a=random.choice(AGS)
 h=f"h:{i['sid']}:{'conn_drop'if i['st']==ST_FAIL else 'part_deg'}"
 c=round(min(0.995,0.35+(1-i['rl'])*random.random()+random.random()*0.3),2)
 return{"agt":a,"hyp":h,"conf":c,"fix":f"rtr_{i['sid']}"}
def hmA(i,v,op):
 sign=op if random.random()<0.88 else None
 scr=random.random()>0.82
 return{"op":op,"sign":sign,"scr":scr}
def esL(i,v):
 return"P1"if i["st"]==ST_FAIL else("P2"if v["conf"]<0.4 else"P3")
L=[]
def acc(r):L.append(r)
def runO(ti=150):
 for t in range(ti):
  ic=dtI();op=ocr(t)
  for i in ic:
   v=agI(i);a=hmA(i,v,op);acc({"t":t,"ic":i,"v":v,"a":a,"l":esL(i,v)})
FM_HEADLESS="HEADLESS :: Humans Endorse Agent Decisions Lacking Evidence Scrutiny Systematically -- the design trades real comprehension for signed tickets, so accountability survives while understanding silently dies; bad agent verdicts get stamped by tired humans who never read them."
def smm():
 p1=sum(1 for x in L if x["l"]=="P1");tot=len(L);st=sum(1 for x in L if x["a"]["sign"]);ns=sum(1 for x in L if not x["a"]["scr"])
 print("inc",tot,"p1",p1,"signed",st,"blind_sign",ns)
 print("FM:",FM_HEADLESS)
 return L
if __name__=="__main__":
 runO();smm()
