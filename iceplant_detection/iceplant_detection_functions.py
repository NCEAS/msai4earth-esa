def iceplant_counts(df):
    return df.filter(items=['iceplant']).groupby(['iceplant']).size().reset_index(name='count')

def test_train_from_df(df,test_size=0.3):
    # Labels are the values we want to predict
    labels = np.array(df['iceplant'])
    #Convert to numpy array
    features = np.array(df.drop('iceplant', axis = 1))
    return train_test_split(features, labels, test_size = test_size, random_state = 42)
import numpy as np
from sklearn.model_selection import train_test_split

# --- check proportions of ice plant vs no ice plant in train/test sets
def test_train_proportions(train_labels, test_labels):
    print('TRAIN SET: Iceplant / no iceplant counts')
    unique, counts = np.unique(train_labels,return_counts=True)
    print(np.column_stack((unique,counts)), '\n')

    print('TEST SET: Iceplant / no iceplant counts')
    unique, counts = np.unique(test_labels,return_counts=True)
    print(np.column_stack((unique,counts)))
    print()
    return

# --- check shapes of train/test features/labels
def  train_test_shapes(train_features, train_labels, test_features, test_labels):
    print('Training Features Shape:', train_features.shape) 
    print('Training Labels Shape:', train_labels.shape) 
    print('Testing Features Shape:', test_features.shape) 
    print('Testing Labels Shape:', test_labels.shape)
    print()
    return

# **********************************************************************************************************

def print_accuracy_info(test_labels,predictions_class):
    N = test_labels.shape[0]
    results = confusion_matrix(test_labels,predictions_class, normalize = 'true')
    
    confmtx = confusion_matrix(test_labels,predictions_class)
    print('true negatives:', confmtx[0,0], 
          '    false positives:', confmtx[0,1])
    print('false negatives:', confmtx[1,0], 
          '    true positives:', confmtx[1,1])
    print()
    unique, counts = np.unique(test_labels,return_counts=True)

    print('true negative %:', np.round(confmtx[0,0]/counts[0]*100,2))
    print('true positive rate %:', np.round(confmtx[1,1]/counts[1]*100,2))
    print()
    print('accuracy %:', np.round( (confmtx[1,1] + confmtx[0,0])/test_labels.shape[0]*100,2))
    return

def plot_roc(rfc, test_features, test_labels):
    ax = plt.gca()
    rf_disp = RocCurveDisplay.from_estimator(rfc, test_features, test_labels, ax=ax)
    return

def print_abs_errors(predictions_class, test_labels):# Calculate the absolute errors
    errors_class = abs(predictions_class - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors_class), 2))
    return

def print_rfc_evaluation(rfc, test_features, test_labels, predictions):
    print_abs_errors(predictions, test_labels)
    print()
    print_accuracy_info(test_labels,predictions)
    plot_roc(rfc, test_features, test_labels)
    print()
    return

def open_window_in_scene(itemid, reduce_box):
    # accesing Azure storage using pystac client
    URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(URL)

    itemid = itemid,
    search = catalog.search(
        collections=["naip"],
        ids = itemid
    )
    item = list(search.get_items())[0]
    # sign and open item
    href = pc.sign(item.assets["image"].href)
    ds = rasterio.open(href)


    reduce = gpd.GeoDataFrame({'geometry':[reduce_box]}, crs="EPSG:4326")
    reduce = reduce.to_crs(ds.crs)

    win = ds.window(*reduce.total_bounds)
    subset = rasterio.open(href).read([1,2,3,4], window=win)
    return subset


def predict_over_subset(itemid, reduce_box,rfc):
    subset = open_window_in_scene(itemid, reduce_box)
    # reshape image into a np.array where each row is a pixel and the columns are the bands
    pixels = subset.reshape([4,-1]).T
    predictions_class = rfc.predict(pixels)
    # turn back into original raster dimensions
    return predictions_class.reshape([subset.shape[1],-1])

# **********************************************************************************************************
# image is a (4,m,n) np array in which bands are r,g,b,nir

def select_ndvi_df(image,thresh=0.2):
    # reshape image into a np.array where each row is a pixel and the columns are the bands
    pixels = image.reshape([4,-1]).T
    df = pd.DataFrame(pixels, columns=['r','g','b','nir'])
    df['ndvi']=(df.nir.astype('int16') - df.r.astype('int16'))/(df.nir.astype('int16') + df.r.astype('int16'))
    vegetation = df[df.ndvi>thresh]
    vegetation.drop(labels=['ndvi'],axis=1, inplace=True)
    return vegetation

# ---------------------------------
def df_backto_image(image, df):
    reconstruct = np.zeros((image.shape[1],image.shape[2]))
    for n in df.index:
        if df.prediction[n]==1:
            i = int((n)/reconstruct.shape[1])
            j = (n) % reconstruct.shape[1]
            reconstruct[i][j] = 1
    return reconstruct

# ---------------------------------
def mask_ndvi_and_predict(itemid, reduce_box, rfc):
    image = open_window_in_scene(itemid, reduce_box)
    veg = select_ndvi_df(image)
    index = veg.index
    features = np.array(veg)
    predictions_class = rfc.predict(features)
    c = {'prediction':predictions_class}
    predictions_df = pd.DataFrame(c, index = index)
    
    return df_backto_image(image,predictions_df)

# **********************************************************************************************************
def plot_window_in_scene(itemid, reduce_box):
    # accesing Azure storage using pystac client
    URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(URL)

    itemid = itemid,
    search = catalog.search(
        collections=["naip"],
        ids = itemid
    )
    item = list(search.get_items())[0]
    # sign and open item
    href = pc.sign(item.assets["image"].href)
    ds = rasterio.open(href)
    
    reduce = gpd.GeoDataFrame({'geometry':[reduce_box]}, crs="EPSG:4326")
    reduce = reduce.to_crs(ds.crs)
    win = ds.window(*reduce.total_bounds)
    
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(np.moveaxis(rasterio.open(href).read([1,2,3], window=win),0,-1))
    plt.show()
    
    return

# image is a (4,m,n) np array in which bands are r,g,b,nir

def select_ndvi_df(image,thresh=0.2):
    # reshape image into a np.array where each row is a pixel and the columns are the bands
    pixels = image.reshape([4,-1]).T
    df = pd.DataFrame(pixels, columns=['r','g','b','nir'])
    df['ndvi']=(df.nir.astype('int16') - df.r.astype('int16'))/(df.nir.astype('int16') + df.r.astype('int16'))
    vegetation = df[df.ndvi>thresh]
    vegetation.drop(labels=['ndvi'],axis=1, inplace=True)
    return vegetation


# ---------------------------------

def select_ndvi_image(itemid, reduce_box):
    subset = open_window_in_scene(itemid, reduce_box)
    df = select_ndvi_df(subset)
    reconstruct = np.zeros((subset.shape[1],subset.shape[2]))
    for n in df.index:
        i = int((n)/reconstruct.shape[1])
        j = (n) % reconstruct.shape[1]
        reconstruct[i][j] = 1
    return reconstruct