from pystyle import Add, Center, Anime, Colors, Colorate, Write, System



def remplacement (nom, remplace, fichier_lecture, fichier_modif): #pour remplacer les varible de la feuille imprimable 
    try:
        file = open(f"{fichier_lecture}","r+")
        a = str(file.read())
        file.close()
        file = open(f"{fichier_modif}","w")

        a = a.replace(f'{nom}', f'{remplace}')
        file.write(a)
        file.close()
        Write.Print("  Remplacement reussi  !", Colors.green, interval=0.000)
        print('')
    except:
        Write.Print("!! Remplacement echoué  !!", Colors.red, interval=0.000)
        print('')
    


df = 'forex_df_verif'
df_live = 'forex_df_live'
squelette1 ='squelette_forex'
squelette2 = 'squelette_forex_prox'


file_obj = open(f"assets/{df}.txt", "r")
file_data = file_obj.read()
lines = file_data.splitlines()
file_obj.close()
dftxt=[]
for argument in lines:
    dftxt.append(argument)


file_obj3 = open(f"assets/{df_live}.txt", "r")
file_data3 = file_obj3.read()
lines3 = file_data3.splitlines()
file_obj3.close()
dftxt3=[]
for argument3 in lines3:
    dftxt3.append(argument3)


le_chemin = '/home/ubuntu/Desktop/admatvforex2/'
#le_chemin = '/home/mat/Bureau/vitesse/'

compteur_launch1 = 0
compteur_launch2 = 0
remplacement('%modif%', f'  python3 {le_chemin}%launcher%', 'assets/launcher2.sh', f'out/launcher.sh')


compteur_launch1 = 0
compteur_launch2 = 0
remplacement('%modif%', f'%launcher%', 'assets/launcher2.py', f'out/launcher.py')

#----- lancement des fonctions remplacements -----#
compte = 0
for arguments in dftxt:

	remplacement('%{squelette}%', f'{arguments}', f'assets/{squelette1}.txt', f'out/{arguments}.py')
	remplacement('%{squelette2}%', f'{dftxt3[compte]}', f'out/{arguments}.py', f'out/{arguments}.py')


	if compteur_launch1 >= 99:
		compteur_launch1 = 0
		remplacement('%launcher%', f'{arguments}.py &\n  sleep 6\n  python3 {le_chemin}%launcher%', 'out/launcher.sh',f'out/launcher.sh')
	if compteur_launch1 < 99:
		remplacement('%launcher%', f'{arguments}.py &\n  python3 {le_chemin}%launcher%', 'out/launcher.sh',f'out/launcher.sh')
	compteur_launch1 = compteur_launch1 + 1

	

	for i in range (0,compteur_launch2):
		remplacement('%launcher%', f'        th{i}.join\n%launcher%', 'out/launcher.py',f'out/launcher.py')

	compte = compte +1

    
    

#----- lancement des fonctions remplacements -----#
