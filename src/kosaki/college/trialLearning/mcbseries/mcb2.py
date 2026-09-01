#一応baiting.pyのhtml連携バージョン
import cgi, sys, io
def main():
    form = cgi.FieldStorage()
    m0 = form.getfirst('whichtype')
    m1 = form.getfirst('hourlypay')
    m2 = form.getfirst('yearlyincome')
    m3 = form.getfirst('money')
    m4 = form.getfirst('time')
    m5 = form.getfirst('day')
    m6 = form.getfirst('overtime')
    m7 = m3*m4*m6*365
    m8 = m2/12/30/24/(m5+m6)
    if m0 == "年収":
        m9 = m0
    elif m0 == "時給":
        m9 = m0
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
    print("Content-type: text/html; charset=UTF-8")
    print("")
    print(mkhead()+str(m0)+mkmiddle1()+str(m1,m4,m5)+mkmiddle2()+str(m2,m3,m6)+mktail()+str(m7,m8)+str(m9))  
def mkhead():
    return '''\
<!DOCTYPE html>
<html lang="ja">
    <head>
        <meta charset="utf-8"/>
        <title>mcb2</title>
    </head>
    <body>
'''
if __name__ == '__main__': 
    main()

def mkmiddle1():
    return '''\
<!DOCTYPE html>
<html lang="ja">
    <head>
        <meta charset="utf-8"/>
        <title>mcb2</title>
    </head>
    <body>
'''
if __name__ == '__main__': 
    main()

def mkmiddle2():
    return '''\
<!DOCTYPE html>
<html lang="ja">
    <head>
        <meta charset="utf-8"/>
        <title>mcb3</title>
    </head>
    
'''
if __name__ == '__main__': 
    main()

def mktail(): 
    return '''\
<!DOCTYPE html>
<html lang="ja">
    <head>
        <meta charset="utf-8"/>
        <title>mcb3</title>
    </head>
    
'''
if __name__=='__main__':
    main()