# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:26:57 2020

@author: lenovo
4650"""
import requests
from lxml import etree
import csv
url="http://huanglab.phys.hust.edu.cn/pepbdb/browse.php"
pep_ids=[]
for i in range(1,266):
    surl=url+"?pagenum="+str(i)
    html=requests.get(surl).text
    selector=etree.HTML(html)
    for j in range(2,52):
        pdbid=selector.xpath("/html/body/div[2]/table/tr["+str(j)+"]/td[1]/a/text()")[0]
        pdbid=str.lower(pdbid)
        peptideid=selector.xpath("/html/body/div[2]/table/tr["+str(j)+"]/td[2]/text()")[0]
        pep_id=pdbid+'_'+peptideid[1]
        pep_ids.append(pep_id)
surl=url+"?pagenum=266"
html=requests.get(surl).text
selector=etree.HTML(html)
for j in range(2,51):
    pdbid=selector.xpath("/html/body/div[2]/table/tr["+str(j)+"]/td[1]/a/text()")[0]
    pdbid=str.lower(pdbid)
    peptideid=selector.xpath("/html/body/div[2]/table/tr["+str(j)+"]/td[2]/text()")[0]
    pep_id=pdbid+'_'+peptideid[1]
    pep_ids.append(pep_id)
print(len(pep_ids))
with open('crawl_results.csv','w',newline="",encoding="utf-8-sig") as f:
    writer=csv.writer(f)
    writer.writerow(["Peptide ID","Interacting peptide residues","Peptide sequence","Interacting receptor residues","Receptor sequence(s)"])
    for pep_id in pep_ids:
            if pep_id=='6mk1_Z':
                continue
            row=[]
            url= "http://huanglab.phys.hust.edu.cn/pepbdb/db/"+pep_id+"/"
            html=requests.get(url).text
            selector=etree.HTML(html)
            ipr=selector.xpath("/html/body/div[2]/table[1]/tr/td/text()")[0]
            ps=selector.xpath("/html/body/div[2]/table[3]/tr/td/text()")
            irr=selector.xpath("/html/body/div[2]/table[2]/tr/td/text()")
            rs=selector.xpath("/html/body/div[2]/table[4]/tr/td/text()")
            irrdict={}
            for item in irr:
                item=item.split(': ')
                irrdict[item[0]]=item[1]
            rsdict={}
            string=''
            for item in rs:
                string+=item
            seq_list=string.split('>')
            seq_list.remove('')
            
            rsdict={}
            for seq in seq_list:
                rsdict[seq[0]]=seq[1:]
            
            for pid in irrdict:
                irr=pid+': '+irrdict[pid]
                rs=pid+': '+rsdict[pid]
                pepid=pep_id+'_'+pid
                row.append(pepid)
                row.append(ipr)
                psfinal=ps[0].strip('>')+': '+ps[1]
                row.append(psfinal)
                row.append(irr)
                row.append(rs)
                writer.writerow(row)
                row.clear()
print("down")
        
    
    