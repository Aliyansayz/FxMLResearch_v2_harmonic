class linear_regression_channel(stochastic_momentum_index):

    def slope_intercept_func(self, batch):
        return np.polyfit(np.arange(len(batch)), batch, 1)

    def deviation_func(self, batch):
        slope, intercept = np.polyfit(np.arange(len(batch)), batch, 1)
        dev = np.sqrt(np.mean((batch - (slope * np.arange(len(batch)) + intercept)) ** 2))
        return dev

    def get_linear_regression_channel_lookback(self, bar_list, period=21, dev_multiplier=2.0, lookback=10):

        symbols, values = 0, 1

        channel_list = [[]] * len(bar_list[values])
        trend_outofchannel_list = [[]] * len(bar_list[values])

        for index, ohlc in enumerate(bar_list[values]):

            open, high, low, close = ohlc['Open'], ohlc['High'], ohlc['Low'], ohlc['Close']
            price = (high + low + close) / 3
            slope, intercept, endy, dev, mid = self.linear_regression_channel(price, period)
            trend_label, outofchannel = self.trend_out_channel_status(price, slope, endy, dev, dev_multiplier)

            if lookback:
                slope, intercept, endy, dev, mid = slope[-lookback:], intercept[-lookback:], endy[-lookback:], dev[
                                                                                                               -lookback:], mid[
                                                                                                                            -lookback:]
                trend_label, outofchannel = trend_label[-lookback:], outofchannel[-lookback:]

            regression_channel = [[slope[i], intercept[i], endy[i], dev[i], mid[i]] for i in range(len(slope))]
            channel_list[index] = regression_channel
            trend_label_outofchannel_status = [[trend_label[i], outofchannel[i]] for i in range(len(trend_label))]
            trend_outofchannel_list[index] = trend_label_outofchannel_status

        return channel_list, trend_outofchannel_list

    def trend_out_channel_status(self, price, slope, endy, dev, dev_multiplier=2.0):

        diff_slope = slope - np.roll(slope, 1)

        labels = np.where(slope > 0,
                          np.where(diff_slope > 0, "uptrend_increasing", "uptrend"),
                          np.where(slope < 0,
                                   np.where(diff_slope < 0, "downtrend_decreasing", "downtrend"),
                                   np.where(diff_slope == 0, "NoTrend", "flat_trend")))

        outofchannel = np.where((slope > 0) & (price < endy - dev * dev_multiplier), "-1 lower breakout",
                                # Lower breakout
                                np.where((slope < 0) & (price > endy + dev * dev_multiplier), "1 upper breakout",
                                         # Upper breakout
                                         "0 no breakout"))  # No breakout

        return labels, outofchannel

    def linear_regression_channel(self, arr, period):

        slope = [[]] * (len(arr) - period + 1)
        intercept = [[]] * (len(arr) - period + 1)
        dev = [[]] * (len(arr) - period + 1)
        endy = [[]] * (len(arr) - period + 1)

        for i in range(len(arr) - period + 1):
            batch = arr[i:i + period]

            slope[i] = self.slope_intercept_func(batch)[0]
            intercept[i] = self.slope_intercept_func(batch)[1]
            endy[i] = intercept[i] + slope[i] * (period - 1)
            dev[i] = self.deviation_func(batch)

        mid = np.convolve(arr, np.ones(period) / period, mode='valid')
        channel_values = [slope, intercept, endy, dev, mid]

        # Add up missing values as average
        window = period - 1

        for index, arr in enumerate(channel_values):
            fixed_arr = np.empty(window + len(arr), dtype=float)
            fixed_arr[:window] = np.nan * window
            fixed_arr[window:] = arr
            fixed_arr[np.isnan(fixed_arr)] = np.nanmean(arr[:period])
            channel_values[index] = fixed_arr

        slope, intercept, endy, dev, mid = channel_values[0], channel_values[1], channel_values[2], channel_values[3], \
                                           channel_values[4]

        slope, intercept, endy, dev, mid = np.round(slope, 2), np.round(intercept, 2), np.round(endy, 2), np.round(dev,
                                                                                                                   2), np.round(
            mid, 2)

        return slope, intercept, endy, dev, mid