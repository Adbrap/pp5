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
    
############## ACTIVATION DES PROXIES ET CRYPTO ET LAUNCHER ##############
prox_on = False
crypto_on = False
Triangle = False
launcher_sh = True
##########################################################################
api_key = '1KsqKOh1pTAJyWZx6Qm9pvnaNcpKVh_8'
#api_key = 'q5li8Y5ldvlF7eP8YI7XdMWbyOA3scWJ'
	
if crypto_on == True:
	df = 'crypto_df2'
	squelette1 ='squelette_titre3000'
	squelette2 = 'squelette_crypto_prox'
if crypto_on == False:
	df = 'titre_df_verif2'
	squelette1 ='squelette_titre3000'
	squelette2 = 'squelette_titre_prox'
	
if Triangle == True:
	squelette1 ='squelette_tri'

file_obj = open(f"assets/{df}.txt", "r")
file_data = file_obj.read()
lines = file_data.splitlines()
file_obj.close()
dftxt=[]
for argument in lines:
    dftxt.append(argument)

if prox_on == True:
	compteur_prox1 = 0
	compteur_prox2 = 0

	prox = open("assets/prox_df.txt", "r")
	prox_data = prox.read()
	lines2 = prox_data.splitlines()
	prox.close()
	dftxt2=[]
	for argument2 in lines2:
    		dftxt2.append(argument2)

le_chemin = '/home/ubuntu/Desktop/3000/'
#le_chemin = '/home/ubuntu/Desktop/vitesse/'
if launcher_sh == True:
	compteur_launch1 = 0
	compteur_launch2 = 0
	compteur_launch3 = 0
	remplacement('%modif%', f'  python3 {le_chemin}%launcher%', 'assets/launcher2.sh', f'out/launcher.sh')
	remplacement('%modif%', f'  python3 {le_chemin}%launcher%', 'assets/launcher2.sh', f'out/launcher2.sh')

if launcher_sh == False:
	compteur_launch1 = 0
	compteur_launch2 = 0
	remplacement('%modif%', f'%launcher%', 'assets/launcher2.py', f'out/launcher.py')

#----- lancement des fonctions remplacements -----#
compteur_api = 1
for arguments in dftxt:

	if prox_on == True:
		if compteur_prox1 >= 10:
			compteur_prox1 = 0
			compteur_prox2 = compteur_prox2 + 1
		remplacement('%{squelette}%', f'{arguments}', f'assets/{squelette2}.txt', f'out/{arguments}.py')
		remplacement('%{proxies}%', f'{dftxt2[compteur_prox2]}', f'out/{arguments}.py', f'out/{arguments}.py')
		compteur_prox1 = compteur_prox1+1

	if prox_on == False:
		remplacement('%{squelette}%', f'{arguments}', f'assets/{squelette1}.txt', f'out/{arguments}.py')
		if compteur_api < 4758:
			remplacement('%{api}%', f'{api_key}', f'out/{arguments}.py', f'out/{arguments}.py')
		if compteur_api >=4758:
			api_key = 'q5li8Y5ldvlF7eP8YI7XdMWbyOA3scWJ'
			remplacement('%{api}%', f'{api_key}', f'out/{arguments}.py', f'out/{arguments}.py')
		
	if launcher_sh == True:
		if compteur_api < 4758:
			if compteur_launch1 >= 75:
				compteur_launch1 = 0
				remplacement('%launcher%', f'{arguments}.py &\n  sleep 10\n  python3 {le_chemin}%launcher%', 'out/launcher.sh',f'out/launcher.sh')
			if compteur_launch1 < 75:
				remplacement('%launcher%', f'{arguments}.py &\n  python3 {le_chemin}%launcher%', 'out/launcher.sh',f'out/launcher.sh')
			compteur_launch1 = compteur_launch1 + 1
			
		if compteur_api >= 4758:
			if compteur_launch3 >= 75:
				compteur_launch3 = 0
				remplacement('%launcher%', f'{arguments}.py &\n  sleep 10\n  python3 {le_chemin}%launcher%', 'out/launcher2.sh',f'out/launcher2.sh')
			if compteur_launch3 < 75:
				remplacement('%launcher%', f'{arguments}.py &\n  python3 {le_chemin}%launcher%', 'out/launcher2.sh',f'out/launcher2.sh')
			compteur_launch3 = compteur_launch3 + 1
	if launcher_sh == False:
		remplacement('%launcher%', f'        th{compteur_launch2}=Process(target=exec(open(\"{le_chemin}{arguments}.py\").read()))\n%launcher%', 'out/launcher.py',f'out/launcher.py')
		compteur_launch2 = compteur_launch2 + 1
	compteur_api = compteur_api +1
if launcher_sh == False:
	remplacement('%launcher%', f'        \n%launcher%', 'out/launcher.py', f'out/launcher.py')
	for i in range (0,compteur_launch2):
		if compteur_launch1 >= 100:
			compteur_launch1 = 0
			remplacement('%launcher%', f'        time.sleep(6)\n%launcher%', 'out/launcher.py',f'out/launcher.py')
		remplacement('%launcher%', f'        th{i}.start\n%launcher%', 'out/launcher.py',f'out/launcher.py')
		compteur_launch1 = compteur_launch1 + 1
	remplacement('%launcher%', f'        \n%launcher%', 'out/launcher.py', f'out/launcher.py')
	

	for i in range (0,compteur_launch2):
		remplacement('%launcher%', f'        th{i}.join\n%launcher%', 'out/launcher.py',f'out/launcher.py')


    
    

#----- lancement des fonctions remplacements -----#
