import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import mixture
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.model_selection import GridSearchCV
from shapely.ops import unary_union, polygonize
import math
import shapely.geometry as geometry
from scipy.spatial import Delaunay
from joblib import dump, load
import geopandas as gpd
import folium
import json
from shapely.geometry import shape, GeometryCollection, Point


class ArgoML():
    
    def __init__(
        self,
        filename: str = 'temperature.csv'
        ):

        self.filename = filename
        self.df = pd.read_csv(f'../raw_data/{self.filename}')
        self.pres = self.df.columns[3:].values
        self.lon = self.df.iloc[:,0].values
        self.lat = self.df.iloc[:,1].values
        self.X = self.df.iloc[:,3:].values
        self.n_components = None
        
    def load_models(self, filename=None):
        if not filename:
            filename = self.filename
        self.pca = load(f'pca_{filename}.joblib') 
        self.model = load(f'gmm_{filename}.joblib')
      
    def scale_data(self):
        self.Xscaled = preprocessing.scale(self.X)

    def pca_analysis(self, save_model=True):
        if self.n_components:
            self.pca = PCA(n_components=self.n_components)
        else:
            self.pca = PCA()

        self.pca.fit(self.Xscaled)
        self.Xpca = self.pca.transform(self.Xscaled)
        if not self.n_components:
            self.n_components = 0
            for i in self.pca.explained_variance_ratio_.cumsum():
                self.n_components += 1
                if i > 0.997:
                    break
                self.pca_analysis()

        if save_model:
            dump(self.pca, f'pca_{self.filename}.joblib') 

    def pca_transform(self):
        self.Xpca = self.pca.transform(self.Xscaled)

    def model_fit(self, show_results=False, save_model=True):
        def gmm_bic_score(estimator, X):
            return -estimator.bic(X)

        model = mixture.GaussianMixture()
        param_grid = {
            "n_components": range(2, 15),
            "covariance_type": ["full"],
            }
        self.grid_search = GridSearchCV(
            model, param_grid=param_grid, scoring=gmm_bic_score, verbose=2
            )
        self.grid_search.fit(self.Xpca)
        self.model = self.grid_search.best_estimator_
        self.model.fit(self.Xpca)

        if show_results:
            self.show_model_result()
        if save_model:
            dump(self.model, f'gmm_{self.filename}.joblib') 
    
    def model_predict(self):
        labels = self.model.predict(self.Xpca)
        posterior_probs = self.model.predict_proba(self.Xpca)
        max_posterior_probs = np.max(posterior_probs,axis=1) 
        self.df.insert(2,'label',labels,True)
        self.df.insert(3,'max_posterior_prob',max_posterior_probs,True) 
        grouped_unsorted = self.df.groupby('label')
        df_means = grouped_unsorted.agg({k: 'first' if k == 'Time' else np.mean for k in self.df.columns})
        self.n_comp = self.model.n_components
        T15_means = df_means['15.0'].values
        old2new = np.argsort(T15_means)
        di = dict(zip(old2new,range(0,self.n_comp)))
        grouped = self.df.groupby('label')
        self.dfg_means = grouped.agg({k: 'first' if k == 'Time' else np.mean for k in self.df.columns})
        self.dfg_stds = grouped.agg({k: 'first' if k == 'Time' else np.std for k in self.df.columns})
        self.nprofs = grouped['Latitude'].count().values

    def model_plot(self):

        p = np.asarray([float(i) for i in self.pres])

        plt.figure(figsize=(35, 42))
        plt.style.use('seaborn-darkgrid')

        # create a color palette
        palette = cm.coolwarm(np.linspace(0,1,self.n_comp))

        # iterate over groups
        num = 0
        for nrow in range(0,self.n_comp):
            num += 1
            mean_lon = self.dfg_means.iloc[nrow,0]
            mean_lat = self.dfg_means.iloc[nrow,1]
            mean_T = self.dfg_means.iloc[nrow,5:].values
            std_T = self.dfg_stds.iloc[nrow,5:].values
            
            # select subplot
            plt.subplot(int(np.ceil(self.n_comp/5)),5,num)
            plt.plot(mean_T, p, marker='', linestyle='solid', color=palette[nrow], linewidth=6.0, alpha=0.9)
            plt.plot(mean_T+std_T, p, marker='', linestyle='dashed', color=palette[nrow], linewidth=6.0, alpha=0.9)
            plt.plot(mean_T-std_T, p, marker='', linestyle='dashed', color=palette[nrow], linewidth=6.0, alpha=0.9)
            
            # custom grid and axes
            plt.ylim([0,1000])
            ax = plt.gca()
            ax.invert_yaxis() 
            ax.grid(True)
            
            fs = 16 # font size
            plt.ylabel('Pressure (dbar)', fontsize=fs)
            plt.title('Class = ' + str(num), fontsize=fs)
            mpl.rc('xtick', labelsize=fs)     
            mpl.rc('ytick', labelsize=fs)
            
            # text box
            textstr = '\n'.join((
                r'N profs. = %i' % (self.nprofs[nrow], ),
                r'Mean lon = %i' % (mean_lon, ),
                r'Mean lat = %i' % (mean_lat, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            # place a text box in upper left in axes coords
            ax.text(0.45, 0.25, textstr, transform=ax.transAxes, fontsize=fs,
                    verticalalignment='top', bbox=props)
        plt.show()

    def create_polygons(self):
        for nrow in range(0,self.n_comp):
            mean_T = self.dfg_means.iloc[nrow,5:].values
			
            std_T = self.dfg_stds.iloc[nrow,5:].values
            points = []
            for i, value in enumerate(mean_T):
                if i == 0:
                    points.append((value-(3*std_T[i]), float(0)))
                    points.append((value+(3*std_T[i]), float(0)))                    
                points.append((value-(3*std_T[i]), float(self.pres[i])))
                points.append((value+(3*std_T[i]), float(self.pres[i])))
                points_point = [geometry.Point(xy) for xy in points]
            concave_hull, edge_points = ArgoML.alpha_shape(points_point, alpha=0.1)
            gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[concave_hull])
            gdf.to_file(f'{nrow}_{self.filename}.json', driver="GeoJSON")

    def plot_folium_map(self):
        colors = {'1': 'green', '2': 'red', '3': 'blue', '4': 'orange', '5': 'black', '6': 'purple', '7': 'yellow', '8': 'gray'}
        folium_map = folium.Map(location=[20,0], tiles="OpenStreetMap", zoom_start=2)
        for nrow in range(0,self.n_comp):
            new_df = self.df[self.df['label'] == nrow]
            for row in new_df.iterrows():
                folium.Marker([row[1]['Latitude'],row[1]['Longitude']], popup='',icon=folium.Icon(color=colors[str(nrow+1)])).add_to(folium_map)
        
        folium_map.save(f"{self.filename}.html")
        folium_map.show_in_browser()

    def show_model_result(self):
        comp_df = pd.DataFrame(self.grid_search.cv_results_)[
            ["param_n_components", "param_covariance_type", "mean_test_score"]
        ]
        comp_df["mean_test_score"] = -comp_df["mean_test_score"]
        comp_df = comp_df.rename(
            columns={
                "param_n_components": "Number of components",
                "param_covariance_type": "Type of covariance",
                "mean_test_score": "BIC score",
            }
        )
        print(comp_df.sort_values(by="BIC score").head())
        sns.catplot(
            data=comp_df,
            kind="bar",
            x="Number of components",
            y="BIC score",
            hue="Type of covariance",
        )
        plt.show()
        
    def open_polygons(self, filename='temp.csv'):
        self.geoms = []
        for i in range(self.n_comp):
            with open(f"{i}_{filename}.json") as f:
                features = json.load(f)["features"]
            
            geom = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in features])
            self.geoms.append(geom)

    def compare_data_polygon(self):
        self.comparison_df = self.df.iloc[:, 5:]

        self.flags = pd.DataFrame(index=self.comparison_df.index,columns=self.comparison_df.columns)
        self.flags[:] = 0
        
        labels = self.df.label
        for row in self.comparison_df.iterrows():
            print(row[0]/self.comparison_df.shape[0])
            geom = self.geoms[int(labels.iloc[row[0]])]
            for i in row[1].items():
                if not Point(i[1],float(i[0])).within(geom):
                    self.flags[i[0]].iloc[row[0]] = 1
    
    def plot_polygon_argo(self, idx=0, plot_flag=True, filename=None):
        labels = self.df.label
        geom = self.geoms[labels.iloc[idx]]
        fig, ax = plt.subplots()
        ax.fill(*geom.geoms[0].exterior.xy, alpha=.3)
        ax.plot(self.df.iloc[idx, 5:].values,self.df.iloc[idx, 5:].index.astype('float'), '.k')
        if plot_flag:
            df = self.df.iloc[idx, 5:]
            if filename:
                flags = pd.read_csv(f'../raw_data/{filename}_flags.csv')  
                flag = flags.iloc[idx, 5:]
                df = df[flag[flag<1].index]
            else:
                flag = self.flags.iloc[idx, 5:]
                df = df[flag[flag>0].index]
            ax.plot(df.values,df.index.astype('float'), '.r')
            
        ax.set_ylim(2000, 0)
        plt.show()

    def alpha_shape(points, alpha):
        def add_edge(edges, edge_points, coords, i, j):
            if (i, j) in edges or (j, i) in edges:
                return
            edges.add( (i, j) )
            edge_points.append(coords[ [i, j] ])
        
        if len(points) < 4:
            return geometry.MultiPoint(list(points)).convex_hull
        
        coords = np.array([point.coords[0] for point in points])

        tri = Delaunay(coords)
        edges = set()
        edge_points = []
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.simplices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]

            # Lengths of sides of triangle
            a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
            b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
            c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

            # Semiperimeter of triangle
            s = (a + b + c)/2.0

            # Area of triangle by Heron's formula
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            if area == 0:
                area = 1.000000000154332e-09
            circum_r = a*b*c/(4.0*area)

            # Here's the radius filter.
            #print circum_r
            if circum_r < 1.0/alpha:
                add_edge(edges, edge_points, coords, ia, ib)
                add_edge(edges, edge_points, coords, ib, ic)
                add_edge(edges, edge_points, coords, ic, ia)

        m = geometry.MultiLineString(edge_points)
        triangles = list(polygonize(m))

        return unary_union(triangles), edge_points        

 