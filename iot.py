import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
import hvplot.pandas
from sklearn.calibration import LabelEncoder
import lightgbm as lgb 
# import shap
from sklearn.metrics import mean_absolute_error
# shap.initjs()

class iot:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path,low_memory=False)
        self._lgbm_df = None
        self.model = None


    def hours2timing(self, x):
        if x in [22,23,0,1,2,3]:
            timing = 'Night'
        elif x in range(4, 12):
            timing = 'Morning'
        elif x in range(12, 17):
            timing = 'Afternoon'
        elif x in range(17, 22):
            timing = 'Evening'
        else:
            timing = 'X'
        return timing
    
    def load(self):
        self.df["time"] = pd.to_datetime(self.df["createDate"])
        self.df.info()
        self.df['year'] = self.df['time'].apply(lambda x : x.year)
        self.df['month'] = self.df['time'].apply(lambda x : x.month)
        self.df['day'] = self.df['time'].apply(lambda x : x.day)
        self.df['weekday'] = self.df['time'].apply(lambda x : x.day_name())
        self.df['weekday_numeric'] = self.df['time'].dt.dayofweek
        self.df['weekofyear'] = self.df['time'].apply(lambda x : x.weekofyear)
        self.df['hour'] = self.df['time'].apply(lambda x : x.hour)
        self.df['minute'] = self.df['time'].apply(lambda x : x.minute)
        self.df['second'] = self.df['time'].apply(lambda x : x.second)
        self.df['date'] = self.df['time'].apply(lambda x : x.date())

        print(self.df.head(3))
        self.df['timing'] = self.df['hour'].apply(self.hours2timing)


    def show_density(self):
        # Create a HoloViews Scatter plot for each column
        hum_plot = hv.Distribution(self.df['hum'],label = "humidity").opts(color='blue')
        snd_plot = hv.Distribution(self.df['snd'], label = "sound").opts(color='green')
        temp_plot = hv.Distribution(self.df['temp'], label = "temperature").opts(color='orange')
        light_plot = hv.Distribution(self.df['light'], label = "light").opts(color='red')

        # Set the options for the combined plot
        combined_plot = (hum_plot * snd_plot * temp_plot * light_plot).opts(opts.Distribution(xlabel = 'units', ylabel = 'density', title = 'Distribution',
                                        width=800, height=350, tools=['hover'], show_grid=True))
        
        # Display the combined plot
        hv.save(combined_plot, 'holo.png', fmt='png')
        hv.save(combined_plot, 'holo.html', fmt='html')




    def groupByMonth(self, col):
        return self.df[[col,'month']].groupby('month').agg({col:['mean']})[col]
    

    def groupByWeekday(self,col):
        weekdayDf = self.df.groupby('weekday').agg({col:['mean']})
        weekdayDf.columns = [f"{i[0]}_{i[1]}" for i in weekdayDf.columns]
        weekdayDf['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in weekdayDf.index]
        weekdayDf.sort_values('week_num', inplace=True)
        weekdayDf.drop('week_num', axis=1, inplace=True)
        return weekdayDf
    
    def groupByTiming(self, col):
        timingDf = self.df.groupby('timing').agg({col:['mean']})
        timingDf.columns = [f"{i[0]}_{i[1]}" for i in timingDf.columns]
        timingDf['timing_num'] = [['Morning','Afternoon','Evening','Night'].index(i) for i in timingDf.index]
        timingDf.sort_values('timing_num', inplace=True)
        timingDf.drop('timing_num', axis=1, inplace=True)
        return timingDf
    
    def groupByDate(self, col, startdate, enddate):
        startdate = pd.to_datetime(startdate).date()
        enddate = pd.to_datetime(enddate).date()
        dateDf = self.df[(self.df['date'] >= startdate) & (self.df['date'] <= enddate)]
        dateDf = dateDf.groupby('date').agg({col:['mean']})
        dateDf.columns = [f"{i[0]}_{i[1]}" for i in dateDf.columns]
        return dateDf

    def show_bytiming(self, mode):
        if mode == 1:
            hu = hv.Curve(self.groupByTiming('hum')).opts(title="Hum", color="red", ylabel="Unit")
            sn = hv.Curve(self.groupByTiming('snd')).opts(title="snd", color="blue", ylabel="Unit")
            li = hv.Curve(self.groupByTiming('light')).opts(title="Light", color="yellow", ylabel="Unit")
            te = hv.Curve(self.groupByTiming('temp')).opts(title="Temp", color="green", ylabel="Unit")

            img = (hu+sn+li+te).opts(opts.Curve(xlabel="Timing", width=400, height=300,tools=['hover'],show_grid=True,fontsize={'title':10})).opts(shared_axes=False)
            hv.save(img, 'bytiming.png', fmt='png')
        elif mode == 2:
            hu = hv.Curve(self.groupByWeekday('hum')).opts(title="Hum", color="red", ylabel="Unit")
            sn = hv.Curve(self.groupByWeekday('snd')).opts(title="snd", color="blue", ylabel="Unit")
            li = hv.Curve(self.groupByWeekday('light')).opts(title="Light", color="yellow", ylabel="Unit")
            te = hv.Curve(self.groupByWeekday('temp')).opts(title="Temp", color="green", ylabel="Unit")

            img = (hu+sn+li+te).opts(opts.Curve(xlabel="Weekday", width=400, height=300,tools=['hover'],show_grid=True,fontsize={'title':10})).opts(shared_axes=False)
            hv.save(img, 'byweekday.png', fmt='png')
            hv.save(img, 'byweekday.html', fmt='html')
        elif mode == 3:
            hu = hv.Curve(self.groupByMonth('hum')).opts(title="Hum", color="red", ylabel="Unit")
            sn = hv.Curve(self.groupByMonth('snd')).opts(title="snd", color="blue", ylabel="Unit")
            li = hv.Curve(self.groupByMonth('light')).opts(title="Light", color="yellow", ylabel="Unit")
            te = hv.Curve(self.groupByMonth('temp')).opts(title="Temp", color="green", ylabel="Unit")

            img = (hu+sn+li+te).opts(opts.Curve(xlabel="Month", width=400, height=300,tools=['hover'],show_grid=True,fontsize={'title':10})).opts(shared_axes=False)
            hv.save(img, 'bymonth.png', fmt='png')
            hv.save(img, 'bymonth.html', fmt='html')

    def show_bydate(self, startdate, enddate):
        hu = hv.Curve(self.groupByDate('hum', startdate, enddate)).opts(title="Hum", color="red", ylabel="Unit")
        sn = hv.Curve(self.groupByDate('snd', startdate, enddate)).opts(title="snd", color="blue", ylabel="Unit")
        li = hv.Curve(self.groupByDate('light', startdate, enddate)).opts(title="Light", color="yellow", ylabel="Unit")
        te = hv.Curve(self.groupByDate('temp', startdate, enddate)).opts(title="Temp", color="green", ylabel="Unit")

        img = (hu+sn+li+te).opts(opts.Curve(xlabel="Date", width=400, height=300,tools=['hover'],show_grid=True,fontsize={'title':10})).opts(shared_axes=False)
        hv.save(img, 'bydate.png', fmt='png')
        hv.save(img, 'bydate.html', fmt='html')

        
    def lgbm_train(self,cols=['hum','snd','light','temp'],trg='hum',train_ratio=0.75,valid_ratio=0.05,test_ratio=0.2):
        self.df.set_index('time', inplace=True)
        self.df.index = pd.to_datetime(self.df.index)
        self.df.info()

        _lgbm_df = self.df.select_dtypes(include=[np.number]).resample('H').mean()
        _lgbm_df['weekday'] =   LabelEncoder().fit_transform(pd.Series(_lgbm_df.index).apply(lambda x : x.day_name())).astype(np.int8)
        _lgbm_df['timing'] = LabelEncoder().fit_transform(_lgbm_df['hour'].apply(self.hours2timing)).astype(np.int8)
        self._lgbm_df = _lgbm_df
            #make dataframe for training
        lgbm_df = _lgbm_df[cols]
        tr,vd,te = [int(len(lgbm_df) * i) for i in [train_ratio, valid_ratio, test_ratio]]
        X_train, Y_train = lgbm_df[0:tr].drop([trg], axis=1), lgbm_df[0:tr][trg]
        X_valid, Y_valid = lgbm_df[tr:tr+vd].drop([trg], axis=1), lgbm_df[tr:tr+vd][trg]
        X_test = lgbm_df[tr+vd:tr+vd+te+2].drop([trg], axis=1)
        lgb_train = lgb.Dataset(X_train, Y_train)
        lgb_valid = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)
        #model training
        params = {
            'task' : 'train',
            'boosting':'gbdt',
            'objective' : 'regression',
            'metric' : {'mse'},
            'num_leaves':200,
            'drop_rate':0.05,
            'learning_rate':0.1,
            'seed':0,
            'feature_fraction':1.0,
            'bagging_fraction':1.0,
            'bagging_freq':0,
            'min_child_samples':5
        }
        gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_train, lgb_valid])
        #make predict dataframe
        pre_df = pd.DataFrame()
        pre_df[trg] = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        pre_df.index = lgbm_df.index[tr+vd:tr+vd+te+2]
        self.model = gbm
        return pre_df, gbm, X_train

    def lgbm_predict_Occ(self, X_predict):
        result = pd.DataFrame()
        result['Occupancy'] = self.model.predict(X_predict[['hum', 'snd', 'light', 'temp']], num_iteration=self.model.best_iteration)
        result.index = X_predict.index
        alertflag = False
        for i in range(len(result)):
            if result['Occupancy'][i] > 0.5 and X_predict['Occupancy'][i] == 0:
                alertflag = True
                break
        return alertflag
 
    
    def lgbm_plot(self, trg = 'hum', lgmbForecast_df = None,portion=0.1):
        # Calculate mean absolute error
        lgbm_use_mae = mean_absolute_error(self._lgbm_df[trg][-len(lgbmForecast_df):], lgbmForecast_df[trg])

        # Create the main plot
        main_plot = hv.Curve(self._lgbm_df[trg], label=trg).opts(color='blue') * \
                    hv.Curve(lgbmForecast_df[trg], label='predicted').opts(color='red', title='LightGBM Result')

        # Create the enlarged plot
        enlarged_plot = hv.Curve(self._lgbm_df[trg][-int(len(self._lgbm_df)*portion):], label=trg).opts(color='blue') * \
                        hv.Curve(lgbmForecast_df[trg], label='predicted').opts(color='red', title='LightGBM Result Enlarged')

        # Combine the plots
        combined_plot = (main_plot.opts(legend_position='bottom') + enlarged_plot.opts(legend_position='bottom')) \
                        .opts(opts.Curve(xlabel="Time", width=800, height=300, show_grid=True, tools=['hover'])) \
                        .opts(shared_axes=False).cols(1)

        # Render the plot
        hv.render(combined_plot)
        hv.save(combined_plot, 'lgbm_plot.png', fmt='png')
        hv.save(combined_plot, 'lgbm_plot.html', fmt='html')

if __name__ == "__main__":
    p = iot("testdata.csv")
    p.load()
    p.show_density()
    p.show_bytiming(1)
    p.show_bytiming(2)
    p.show_bytiming(3)
    lgbmForecast_df, model, x_train = p.lgbm_train(cols=['hum', 'snd', 'light', 'temp','Occupancy'], trg = 'Occupancy', train_ratio=0.8,valid_ratio=0.05,test_ratio=0.15)

    # explainer = shap.TreeExplainer(model=model,feature_perturbation='tree_path_dependent')
    # shap_values = explainer.shap_values(X=x_train)
    p.lgbm_plot(trg = 'Occupancy', lgmbForecast_df = lgbmForecast_df)

    # print(p.lgbm_predict_Occ(p._lgbm_df[['hum', 'snd', 'light', 'temp','Occupancy']]))
    # Calculate mean absolute error
    # p.load()
    # p.show_holo()
    # p.show_bydate('2023-04-20', '2023-04-27')      