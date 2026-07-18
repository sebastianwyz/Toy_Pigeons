Il file da runnare e' run_toy(1), viene caricata una mappa dalla cartella bathymetry maps, attualmente in uso una di quelle in formato jpg, ma c'e' anche quella vera in formato .tif .

vengono create delle traiettorie e da queste ne prendiamo 2:
 - quella con profondita' media inferiore viene adoperata come punto di partenza per Pigeons
 - quella con profondita' media piu' alta viene usata per estrapolare il profilo di profondit' (saranno i dati)

L'obiettivo e' vedere se Pigeons e' in grado di trasformare la traiettoria in modo che passi dal canale con profondita' inferiore a quello con profondita' superiore.
Attualmente in uso explorer SliceSampler (Gibbs), nel codice sui dati veri sto usando AutoMALA.
