# Podržano učenje u svrhu market makinga

Prva verzija koristi nasumično kretanje cijena, kao u Avellaneda-Stoikov radu. Model se nalazi u direktoriju *models*. Okoline u kojima se agent može naći nalaze se u direktoriju *envs*. Algoritam Q-učenja nalazi se u *learning*, a pomoćne funkcije za prikaz u *plotting*. U korijenskom direktoriju *main.py* i *demo_notebook* sadrže efektivno istu stvar, samo Jupyter bilježnica omogućava direktan pregled rezultata iz pretraživača.

Kao trenutno najveći problem identificirao bih kako ostvariti dinamiku izvršavanja *limit ordera*. Tu radovi koriste drukčije pristupe, i nisam nijedan uspio u potpunosti ostvariti pa je trenutno vrlo jednostavna verzija implementirana. Nakon toga trebalo bi pokušati dodati CARA utility, jer sam i s njom imao problema zbog prevelike razlike u vrijednostima. 

Od ostalih stvari koje su mi još pale na pamet da bi se mogle u budućnosti dodati:
* Osim klasičnog RL agenta dodati i druge pristupe, kao npr. *zero-tick, random action* i sl.
* Usporediti rezultate s tim pristupima (relativni grafovi, kao u referentnom radu)
* Nakon implementacije CARA-e, ispitati i njen utjecaj
* Dosta parametara je hardkodirano u *environment* razredima, ne bi bilo loše izvući ih kako bi se oni mogli optimirati
* Istražiti još možda druge opcije za grafički prikaz rezultata
