from PIL import Image
from PIL.ExifTags import TAGS

def get_pixel_info(image_file_path):
    """
    Estrae la dimensione lineare e l'area di un singolo pixel
    dai metadati EXIF dell'immagine.

    Parametri:
        image_file_path (str): Percorso al file immagine.

    Ritorna:
        dict: {
            "pixel_size_um": float,
            "pixel_area_um2": float
        } oppure None se i dati non sono disponibili.
    """
    try:
        image = Image.open(image_file_path)
        exif_data = image._getexif()
        if not exif_data:
            print("Nessun dato EXIF trovato.")
            return None

        readable_exif = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
        x_res = readable_exif.get("FocalPlaneXResolution")
        res_unit = readable_exif.get("FocalPlaneResolutionUnit")

        if not x_res or not res_unit:
            print("Dati di risoluzione del piano focale mancanti.")
            return None

        # Calcolo dimensione pixel (lato) in micrometri
        if res_unit == 2:  # Inch
            pixel_size_um = (1 / x_res) * 25.4 * 1000
        elif res_unit == 3:  # Centimetri
            pixel_size_um = (1 / x_res) * 10 * 1000
        else:
            print(f"Unità di misura non supportata (code: {res_unit})")
            return None

        # Calcolo area pixel in micrometri quadrati
        pixel_area_um2 = pixel_size_um ** 2

        return {
            "pixel_size_um": pixel_size_um,
            "pixel_area_um2": pixel_area_um2
        }

    except Exception as e:
        print(f"Errore durante la lettura dell'immagine: {e}")
        return None

# Esempio d'uso
if __name__ == "__main__":
    #print('ciao')
    path = "ausiliaria\\22b.jpg"  # Sostituisci con il tuo percorso immagine
    info = get_pixel_info(path)
    if info:
        print(f"Dimensione pixel: {info['pixel_size_um']:.2f} µm")
        print(f"Area pixel: {info['pixel_area_um2']:.2f} µm²")
    else:
        print("Impossibile determinare le informazioni del pixel.")