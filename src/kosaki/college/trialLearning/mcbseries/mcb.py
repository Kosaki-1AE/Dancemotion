#一応baiting.pyのhtml連携バージョン
import cgi, sys, io
def main():
    form = cgi.FieldStorage()
    m0 = form.getfirst('whichtype')
    m1 = form.getfirst('hourlypay')
    m2 = form.getfirst('yearlyincome')
    if m0 == "年収":
        m9 = m0
    elif m0 == "時給":
        m9 = m0
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
    print("Content-type: text/html; charset=UTF-8")
    print("")
    print(mkhead()+str(m0)+mktail()+str(m9)+"→"+str(m1,m2))  
def mkhead():
    return '''\
<!DOCTYPE html>
<html lang="ja">
    <head>
        <meta charset="utf-8"/>
        <title>mcb</title>
    </head>
    <body>
'''

def mktail(): 
    return '''\
<br/>
    <p><a href="/mcb.html"></a></p>
    <hr/>
    </body>
</html>
'''
if __name__=='__main__':
    main()