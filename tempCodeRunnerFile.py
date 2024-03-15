if alertflag == True:
            main_plot = hv.Curve(self.df['Occupancy'][len(self.df['Occupancy'])-len(lgbmForecast_df):], label='Occupancy').opts(color='blue') * \
            hv.Curve(lgbmForecast_df['Occupancy'], label='predicted').opts(color='red', title='Suspicious Activity Detected')
        else:
            main_plot = hv.Curve(self.df['Occupancy'][len(self.df['Occupancy'])-len(lgbmForecast_df):], label='Occupancy').opts(color='blue') * \
            hv.Curve(lgbmForecast_df['Occupancy'], label='predicted').opts(color='red', title='LightGBM Result')