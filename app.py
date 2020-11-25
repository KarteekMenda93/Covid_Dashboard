### Import Packages
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score
import shap
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import altair as alt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

#import pickle
#from plotly.subplots import make_subplots

### Changing the name to our desired name and with the emoji.
st.set_page_config(page_title = 'Default Prediction', page_icon ='💳')


### For byte encoder
import base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

### for unpickling of the models
#model=pickle.load(open("final_log_model.pkl", "rb"))

## Data operations and return the latest data
@st.cache(persist=True)
def load_data():
    data1 = pd.read_csv("covid_feature engineered.csv")
    data1['Employment..number.of.persons.employed.'] = data1['Employment..number.of.persons.employed.'].fillna(
        data1.groupby(['country', 'Quarter'])['Employment..number.of.persons.employed.'].transform('mean'))
    data1['Volume.index.of.production'] = data1['Volume.index.of.production'].fillna(
        data1.groupby(['country', 'Quarter'])['Volume.index.of.production'].transform('mean'))
    data1['Volume.index.of.production'] = data1['Volume.index.of.production'].fillna(0)
    data1['Trade_Volume'] = data1['Trade_Volume'].fillna(0)
    data1['Stock_close'] = data1['Stock_close'].fillna(0)
    data1['equity_skewness'] = data1['equity_skewness'].fillna(0)
    data1['equity_kurtosis'] = data1['equity_kurtosis'].fillna(0)
    data1.drop(["Id", "Company", "Symbol", "Stock.Exchange", "country", "Currency", "zip_code", "Industry", "NUTS3",
                "Quarter", "Ticker"], axis=1, inplace=True)
    return data1

# Split the data
@st.cache(persist=True)
def split(df):
    X = np.array(df.iloc[:, 0:8])
    Y = np.array(df.iloc[:, 8])
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=0)
    return xtrain, xtest, ytrain, ytest

### Helper function for getting the performance of the model.
def plot_metrics(metrics_list):
    if "Confusion_matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, xtest, ytest, display_labels=class_name)
        st.pyplot()
    if "ROC" in metrics_list:
        st.subheader("ROC")
        plot_roc_curve(model, xtest, ytest)
        st.pyplot()
    if "Precision Recall Curve" in metrics_list:
        st.subheader("Precision Recall Curve")
        plot_precision_recall_curve(model, xtest, ytest)
        st.pyplot()

### Helper function to display all the model relevant outputs
def predict_default(EBIDTA, ROA, OperatingCashflow, GrossProfit, TotalRevenue, EBITDAMargin,TotalOperatingExpense,
                                NetProfitMargin, ReturnonEquity, debttoEBITDA, Workingcapital, ChangesinWorkingCapital,
                                Debtorsdayssales, GDP_per_Capita, GDPinMillionEuro):
    input = np.array([[EBIDTA, ROA, OperatingCashflow, GrossProfit, TotalRevenue, EBITDAMargin, TotalOperatingExpense,
                      NetProfitMargin, ReturnonEquity, debttoEBITDA, Workingcapital, ChangesinWorkingCapital,
                      Debtorsdayssales, GDP_per_Capita, GDPinMillionEuro]])
    prediction = model.predict_proba(input)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)


### Main function
def main():

    ### data collection
    activities = ['Data Collection', 'EDA Plots', 'Feature Importances', 'Default Classification', 'Covid Dashboard']
    choices = st.sidebar.radio("Select Activity", activities)
    if choices == 'Data Collection':
        html_temp = """
            <div style="background-color:teal ;padding:10px">
            <h2 style="color:yellow;text-align:center;">Data Collection</h2>
            </div>
            """
        st.markdown(html_temp, unsafe_allow_html=True)
        image = Image.open('aboutdata.PNG')
        st.image(image, caption='Data Collection', use_column_width = True)
        st.subheader("Glimpse of the data")
        data = pd.read_csv('covid_final_data.csv')
        df = data.head(30)
        st.write(df)

    #For EDA plots
    if choices == 'EDA Plots':
        html_temp = """
                    <div style="background-color:teal ;padding:10px">
                    <h2 style="color:yellow;text-align:center;">EDA Plots</h2>
                    </div>
                    """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.subheader("Proportion of Defaulters and Non-Defaulters")
        data = load_data()
        labels = 'Defaulters', 'Non_Defaulters'
        sizes = [data.Target[data['Target'] == 1].count(), data.Target[data['Target'] == 0].count()]
        explode = (0, 0.1)
        fig1, ax1 = plt.subplots(figsize=(15, 8))
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        st.pyplot()
        st.subheader("25.8% of the industries default and the remaining are non defaulters which is almost like out of every 4 industries, 1 is defaulter.")

    # For Feature Importance
    if choices == 'Feature Importances':
        html_temp = """
                            <div style="background-color:teal ;padding:10px">
                            <h2 style="color:yellow;text-align:center;">Feature Importances</h2>
                            </div>
                            """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.subheader("Top 20 Features which proves to be most useful")
        data2 = load_data()
        y = data2['Target']
        X = data2.drop(columns=['Target'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
        feat_importances.nlargest(20).plot(kind='barh')
        plt.xlabel("Random Forest Feature Importance")
        st.pyplot()

    # For the prediction of the model
    if choices == 'Default Classification':
        html_temp = """             <div style="background-color:teal ;padding:10px">
                                    <h2 style="color:yellow;text-align:center;">Default Classification</h2>
                                    </div>
                                    """
        st.markdown(html_temp, unsafe_allow_html=True)

        ### data entry
        activities = ['Classify', 'Metrics', 'Interpretation of Model']
        choices = st.sidebar.radio("Select Activity", activities)

        #### data Entry from the client
        st.subheader("Enter the parameters")
        first, second, third, fourth, fifth, sixth = st.beta_columns(6)
        EBIDTA = first.text_input('EBIDTA score')
        ROA = second.text_input('ROA')
        OperatingCashflow = third.text_input('Operating.Cashflow')
        GrossProfit = fourth.text_input('Gross.Profit')
        TotalRevenue = fifth.text_input('Total.Revenue')
        EBITDAMargin = sixth.text_input('EBITDA.Margin')

        seventh, eight, ninth, tenth, eleventh, twelveth = st.beta_columns(6)
        TotalOperatingExpense = seventh.text_input('Total.Operating.Expense')
        NetProfitMargin = eight.text_input('Net.Profit.Margin')
        ReturnonEquity = ninth.text_input('Return.on.Equity')
        debttoEBITDA = tenth.text_input('debt.to.EBITDA')
        Workingcapital = eleventh.text_input('Working.capital')
        ChangesinWorkingCapital = twelveth.text_input('Changes.in.Working.Capital')

        Thirteen, forteen, fifteen = st.beta_columns(3)
        Debtorsdayssales = Thirteen.text_input('Debtors.days.sales')
        GDP_per_Capita = forteen.text_input('GDP_per_Capita')
        GDPinMillionEuro = fifteen.text_input('GDP..in.Million.Euro')

        #### The main chunk where we classify the industry
        if choices == 'Classify':
            st.subheader("Classify")
            output = predict_default(EBIDTA, ROA, OperatingCashflow, GrossProfit, TotalRevenue, EBITDAMargin,
                                     TotalOperatingExpense, NetProfitMargin, ReturnonEquity, debttoEBITDA,
                                     Workingcapital, ChangesinWorkingCapital, Debtorsdayssales, GDP_per_Capita,
                                     GDPinMillionEuro)
            st.success('The probability of Industry being defaulted is {}'.format(output))
            st.balloons()
            if output >= 0.5:
                st.markdown(default_html, unsafe_allow_html=True)
            else:
                st.markdown(no_default_html, unsafe_allow_html=True)

        ### Show the metrics for the corresponding model.
        if choices == 'Metrics':
            st.subheader("Metrics")
            metrics = st.sidebar.multiselect("What metrics to plot?",
                                             ("Confusion_matrix", "ROC", "Precision Recall Curve"))
            st.write("Model Accuracy: ", model.score(xtest, ytest))
            st.write("Model Precision: ", precision_score(ytest, ypred, labels=class_name))
            plot_metrics(metrics)

        ### For explainability of our model.
        if choices == "Interpretation of the Model":
            st.subheader("Check for the explainability")
            #### Plotting the SHAP values.
            # explain the model's predictions using SHAP
            # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            # visualize the training set predictions
            st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)

    # For Covid related statistics across the world
    if choices == 'Covid Dashboard':
        html_temp = """
                                            <div style="background-color:teal ;padding:10px">
                                            <h2 style="color:yellow;text-align:center;">Covid 19 Dashboard</h2>
                                            </div>
                                            """
        st.markdown(html_temp, unsafe_allow_html=True)
        # Reading the world data from covid.ourworldindata.org
        # The data is cumulative, so the total_cases for a day sums up the cases per day
        df1 = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/full_data.csv")
        analysis = st.sidebar.selectbox("Analysis", ["Overview", "Fatalities", "Trend"])
        if analysis == "Overview":
            o1 = st.selectbox("Dashboards", ["Global", "India"])
            if o1 == "Global":
                # Getting the unique dates present in the dataset
                dates = list(set(df1.date))
                # Sorting the dates to get the most recent date
                # If we use datetime to get the current day, at 12am there will be no data to show as the data for the day would not be updated yet
                dates.sort()
                dt_tday = dates[-1]
                # Getting the data for the most recent date
                td = df1[df1['date'] == dt_tday]
                # Resetting the index
                td = td.reset_index(drop=True)
                # This the text used for the hover data, anything to be added to it should be done here
                # Add a '<br>' after each name and data to move to the next line
                txt = ' Country: ' + td['location'].astype(str) + '<br>' + ' Cases: ' + td['total_cases'].astype(
                    str) + '<br>' + ' Deaths: ' + td['total_deaths'].astype(str)
                # The country names are converted to lowercase for compatibility with the inbuilt location names in graph_object plotting
                td['location'] = td['location'].str.lower()
                # Saving the world data from the dataset
                world = td[td['location'] == 'world']
                # Removing the world data from the dataset
                td = td[td['location'] != 'world']
                # This is to plot the global map
                fig1 = go.Figure(data=go.Choropleth(
                    locations=td['location'],
                    locationmode='country names',
                    z=td['total_cases'],  # Colour of the countries are based on this value
                    colorbar_title="Total Cases",
                    text=txt,  # Hoverdata
                    colorbar={'len': 0.75, 'lenmode': 'fraction'},
                    marker_line_color='black',
                    marker_line_width=0.5))

                fig1.update_layout(
                    title={  # This is to set the size and location of the title of the map
                        'text': 'Global Covid Data',
                        'y': 1,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'color': 'Black', 'size': 30}},
                    geo=dict(  # Removing the frame borders and giving the projection type
                        showframe=False,
                        showcoastlines=True,
                    ),
                    margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.plotly_chart(fig1, config={'displayModeBar': False})
                # Printing out the statistics of the world for the most recent date
                st.header("World Statistics")
                c = world['total_cases'].iloc[0]
                d = world['total_deaths'].iloc[0]
                st.write("Confirmed Cases:", int(c))
                st.write("Confirmed Deaths:", int(d))
                st.write("Fatality Rate:", round((d / c) * 100, 2), '%')
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.header("Country-wise statistics")
                # Getting a list of all the unique contries present after removing 'World'
                countries = list(set(df1.location))
                countries.remove('World')
                countries.remove('International')
                countries.sort()
                countries.remove('India')
                countries.insert(0, 'India')
                # The dropdown for selecting the country
                option1 = st.selectbox("Country", countries)
                # Checking if there is a country selected and if there is, give its information
                if (len(option1) != 0):
                    # This is to pull out the day and total cases for the selected country
                    day_data = {}
                    temp = df1[df1['location'] == option1]
                    day_data[f'{option1} date'] = temp['date']
                    day_data[f'{option1} cases'] = temp[['total_cases']].diff(axis=0).fillna(0).astype(int)
                    day_data[f'{option1} deaths'] = temp[['total_deaths']].diff(axis=0).fillna(0).astype(int)
                    # Plot used for the Univ.ai Covid dashboard question
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(211)
                    ef = pd.DataFrame()
                    ef['date'] = day_data[f'{option1} date'].astype('datetime64[ns]')
                    ef['cases'] = day_data[f'{option1} cases']
                    ax.bar(ef.date, ef.cases, color='#007acc', alpha=0.3)
                    ax.plot(ef.date, ef.cases, marker='o', color='#007acc')
                    # ax.text(0.01,1,f'{option1} daily case count',transform = ax.transAxes, fontsize = 23);
                    ax.set_title(f'{option1} daily case count', fontsize=23)
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
                    ax.tick_params(rotation=60)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax1 = fig.add_subplot(212)
                    ef['deaths'] = day_data[f'{option1} deaths']
                    ax1.bar(ef.date, ef.deaths, color='#007acc', alpha=0.3)
                    ax1.plot(ef.date, ef.deaths, marker='o', color='#007acc')
                    ax1.set_title(f'{option1} daily death count', fontsize=23)
                    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
                    ax1.tick_params(rotation=60)
                    ax1.spines['right'].set_visible(False)
                    ax1.spines['top'].set_visible(False)
                    fig.tight_layout()
                    st.plotly_chart(fig, config={'displayModeBar': False})
                    # Printing out information for the country for the most recent date
                    c = td[td['location'] == option1.lower()]['total_cases'].iloc[0]
                    d = td[td['location'] == option1.lower()]['total_deaths'].iloc[0]
                    st.write("Confirmed Cases:", int(c))
                    st.write("Confirmed Deaths:", int(d))
                    st.write("Fatality Rate:", round((d / c) * 100, 2), '%')
                df = df1
                df = df[(df.location != 'World') & (df.location != 'International')]
                df['date'] = df['date'].astype('datetime64[ns]')
                st.header('Comparison of infection growth')
                st.write("")
                country_name_input = st.multiselect(
                    'Country name',
                    df.groupby('location').count().reset_index()['location'].tolist())
                # by country name
                if len(country_name_input) > 0:
                    subset_data = df[df['location'].isin(country_name_input)]

                    total_cases_graph = alt.Chart(subset_data).transform_filter(
                        alt.datum.total_cases > 0
                    ).mark_line().encode(
                        x=alt.X('date:T', title='Date', timeUnit='yearmonthdate', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('total_cases:Q', title='Confirmed cases'),
                        color='location',
                        tooltip=['location', 'total_cases'],
                    ).properties(
                        width=750,
                        height=300
                    ).configure_axis(
                        labelFontSize=10,
                        titleFontSize=15
                    )
                    st.altair_chart(total_cases_graph)
            elif o1 == "India":
                # Repeating the same for India
                st.title('Covid Analysis for India')
                # Reading the data from covid19india.org
                data = pd.read_csv("https://api.covid19india.org/csv/latest/states.csv")
                # The data contains an unassigned state which is removed
                df = data[data['State'] != 'State Unassigned']
                # Removing unnecessary columns
                df = df[['Date', 'State', 'Confirmed', 'Recovered', 'Deceased']]
                # Renaming the columns since the hover data is based on the column names
                df.columns = ['Date', 'State', 'Confirmed Cases', 'Recovered', 'Deceased']
                # Getting a list of all dates
                dates = list(set(df.Date))
                dates.sort()
                # Getting a list of all States
                states = list(set(df.State))
                # Findingtodays date
                dt_tday = dates[-1]
                # Finding yesterdays date
                dt_yday = dates[-2]
                # Getting todays data for all states available
                dfc = df[df['Date'] == dt_tday]
                # This is done for compatibility of state names with the geojson
                dfc = dfc.replace("Andaman and Nicobar Islands", 'Andaman & Nicobar')
                # Saving the data for India
                India = dfc[dfc['State'] == 'India']
                # Removing India's data from the dataset
                dfc = dfc[dfc['State'] != 'India']
                # Link to the geojson
                gj = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
                fig2 = px.choropleth(
                    dfc,
                    geojson=gj,
                    featureidkey='properties.ST_NM',
                    locations='State',
                    color='Confirmed Cases',
                    color_continuous_scale="Blues",
                    projection="mercator",
                    hover_data=['State', 'Confirmed Cases', 'Deceased',
                                'Recovered'])  # The data is pulled out from the dataframe dfc in this case
                fig2.update_geos(fitbounds="locations", visible=False)
                fig2.update_layout(
                    autosize=False,
                    width=700,  # Here I am able to change the height and width of the graph unlike before
                    height=700,
                    title={
                        'y': 1,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'color': 'blue', 'size': 40}}
                )
                st.plotly_chart(fig2, config={'displayModeBar': False})
                # Printing out India's stats
                st.header("India Statistics")
                c = India['Confirmed Cases'].iloc[0]
                d = India['Deceased'].iloc[0]
                r = India['Recovered'].iloc[0]
                st.write("Confirmed Cases:", int(c))
                st.write("Confirmed Deaths:", int(d))
                st.write("Recovered:", int(r))
                st.write("Current Cases:", int(c - d - r))
                st.write("Fatality Rate:", round((d / c) * 100, 2), '%')
                st.write("Recovery Rate:", round((r / c) * 100, 2), '%')
                # Removing India from the list of states
                states = list(dfc.sort_values(by=['Confirmed Cases', 'Deceased'], ascending=[False, False]).State)
                option = st.selectbox("State", states)
                # Giving the information for each state similar to info for each country
                if (len(option) != 0):
                    day_data = {}
                    temp = df[df['State'] == option]
                    day_data[f'{option} date'] = temp['Date']
                    day_data[f'{option} cases'] = temp[['Confirmed Cases']].diff(axis=0).fillna(0).astype(int)
                    day_data[f'{option} recovered'] = temp[['Recovered']].diff(axis=0).fillna(0).astype(int)
                    day_data[f'{option} deaths'] = temp[['Deceased']].diff(axis=0).fillna(0).astype(int)
                    fig = plt.figure(figsize=(8, 9))
                    ax = fig.add_subplot(311)
                    ef = pd.DataFrame()
                    ef['date'] = day_data[f'{option} date'].astype('datetime64[ns]')
                    ef['cases'] = day_data[f'{option} cases']
                    ax.bar(ef.date, ef.cases, color='#007acc', alpha=0.3)
                    ax.plot(ef.date, ef.cases, marker='o', color='#007acc')
                    ax.set_title(f'{option} daily case count', fontsize=23);
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
                    ax.tick_params(rotation=60)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax1 = fig.add_subplot(312)
                    ef['deaths'] = day_data[f'{option} deaths']
                    ax1.bar(ef.date, ef.deaths, color='#007acc', alpha=0.3)
                    ax1.plot(ef.date, ef.deaths, marker='o', color='#007acc')
                    ax1.set_title(f'{option} daily death count', fontsize=23)
                    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
                    ax1.tick_params(rotation=60)
                    ax1.spines['right'].set_visible(False)
                    ax1.spines['top'].set_visible(False)
                    ax2 = fig.add_subplot(313)
                    ef['recovered'] = day_data[f'{option} recovered']
                    ax2.bar(ef.date, ef.recovered, color='#007acc', alpha=0.3)
                    ax2.plot(ef.date, ef.recovered, marker='o', color='#007acc')
                    ax2.set_title(f'{option} daily recovery count', fontsize=23)
                    ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
                    ax2.tick_params(rotation=60)
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    fig.tight_layout()
                    st.plotly_chart(fig, config={'displayModeBar': False})
                    dfc = dfc.replace("Andaman & Nicobar", 'Andaman and Nicobar Islands')
                    c = dfc[dfc['State'] == option]['Confirmed Cases'].iloc[0]
                    d = dfc[dfc['State'] == option]['Deceased'].iloc[0]
                    r = dfc[dfc['State'] == option]['Recovered'].iloc[0]
                    st.write("Confirmed Cases:", int(c))
                    st.write("Confirmed Deaths:", int(d))
                    st.write("Recovered:", int(r))
                    st.write("Current Cases:", int(c - d - r))
                    st.write("Fatality Rate:", round((d / c) * 100, 2), '%')
                    st.write("Recovery Rate:", round((r / c) * 100, 2), '%')

        elif analysis == "Fatalities":
            df2 = pd.read_csv(
                "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")

            # getting the latest date in the dataset
            dat = list(set(df2.date))
            dat.sort()
            dat = dat[-1]

            # gives the most recent data of every country
            df2_temp = df2.loc[df2['date'] == dat]
            df2_temp = df2_temp[(df2_temp.location != 'World') & (df2_temp.location != 'International')]
            req_data = df2_temp
            req_data['total_deaths'] = req_data['total_deaths'].fillna(0)
            req_data = req_data.set_index("location")

            # dropping rows with 0 cases
            for loca in req_data.index:
                if (req_data.total_cases[loca] < 1.0):
                    req_data = req_data.drop([loca])

            # fatalities number
            desired = (req_data["total_deaths"].sort_values(ascending=True))

            # fatalities %
            desired2 = desired.copy()
            for i in desired.index:
                desired2[i] = (desired[i] / req_data.loc[i, "total_cases"]) * 100
            desired2 = (desired2.sort_values(ascending=True))
            f1 = st.selectbox("Fatalities", ["By number", "By rate"])
            if f1 == "By number":
                fig = plt.figure(figsize=(10, 15))
                ax = fig.gca()  # get current axes for figure
                desired[-20:].plot.barh(color='r', alpha=0.7)
                for p, c, ch in zip(range(desired[-20:].shape[0]), desired[-20:].index, desired[-20:].values):
                    plt.annotate(str(round(ch)), xy=(ch + 1, p), va='center')
                ax.tick_params(axis="both", which='both',  # major and minor ticks
                               length=0)
                ax.tick_params(axis="x", labeltop=False, labelbottom=False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                st.write(fig)

            elif f1 == "By rate":
                fig = plt.figure(figsize=(10, 15))
                ax = fig.gca()  # get current axes for figure
                desired2[-20:].plot.barh(color='r', alpha=0.7)
                for p, c, ch in zip(range(desired2[-20:].shape[0]), desired2[-20:].index, desired2[-20:].values):
                    plt.annotate(str(round(ch, 1)) + "%", xy=(ch + 0.1, p), va='center')
                ax.tick_params(axis="both", which='both',  # major and minor ticks
                               length=0)
                ax.tick_params(axis="x", labeltop=False, labelbottom=False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                st.write(fig)

        elif analysis == "Trend":
            t1 = st.selectbox("Global cases trend", ["Past week", "Past month"])
            df = df1
            df = df[(df.location != 'World') & (df.location != 'International')]
            trend1 = (df.groupby("date").sum()["total_cases"] / float(1e6))
            if t1 == "Past week":
                fig = plt.figure(figsize=(16, 10))
                ax = fig.gca()
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f M'))  # adding unit to y-axis values
                ax.set_xlabel("Date", fontsize=23)
                ax.set_ylabel("Global Cases", fontsize=23)
                ax.plot(trend1[-8:-1], "ro")
                ax.plot(trend1[-8:-1], alpha=0.2)
                st.write(fig)
            elif t1 == "Past month":
                fig = plt.figure(figsize=(16, 10))
                ax = fig.gca()
                ax.set_xticks([0, 4, 9, 14, 19, 24, 29])
                ax.set_xticklabels(
                    [trend1.index[-31], trend1.index[-26], trend1.index[-21], trend1.index[-16], trend1.index[-11],
                     trend1.index[-6], trend1.index[-2]])
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f M'))
                ax.plot(trend1[-31:-1], "ro")
                ax.plot(trend1[-31:-1], alpha=0.2)
                ax.set_xlabel("Date", fontsize=23)
                ax.set_ylabel("Global Cases", fontsize=23)
                st.write(fig)

if __name__=='__main__':
    main()