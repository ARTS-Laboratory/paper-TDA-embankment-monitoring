import os, tqdm
import pandas as pd
from tda import tda, pca_bg_removal, ransac_tda
from tda_utility import load_laz


def load_laz(file_path, arr=True):
    """
    Reads a .laz point cloud file and returns a numpy array of point coordinates.

    Parameters:
        file_path (str): Path to the .laz file.

    Returns:
        np.ndarray: An array of shape (N, 3) with columns for X, Y, and Z coordinates.
    """
    # Read the file using laspy (compatible with laspy 2.x)
    las = laspy.read(file_path)

    if arr:
        # Extract X, Y, Z coordinates
        x = las.x
        y = las.y
        z = las.z

        # Combine the coordinates into a (N, 3) array
        points = np.column_stack((x, y, z))
        return points
    else:
        return las


def main(root):
    df = pd.DataFrame(columns=['Date', 'H0', 'H1'])
    laz_lst = sorted(file for file in os.listdir(root) if file.endswith('laz'))
    for file in tqdm.tqdm(laz_lst):
        print(file)
        path = os.path.join(root, file)
        pcd = load_laz(path, True)
        print('Removing noise...')
        pcd = pca_bg_removal(pcd, delta=0.6)
        print('Computing TDA...')
        ent = ransac_tda(pcd, dim=1, fts='entropy', k=100, m=1000)
        ent = [file[:-4]] + list(ent[0])
        df.loc[len(df)] = ent
    df.to_csv(root+ '/h02h1.csv')



if __name__ == "__main__":
    root = 'G://Research/Projects/Embankment/SLidE'
    main(root)
