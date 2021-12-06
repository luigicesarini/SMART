library(shiny)
library(ggplot2)
library(tidyverse)
library(gganimate)
library(transformr)
library(tmap)
library(data.table)
library(plotly)
library(sf)
library(sicegar)
library(glue)
library(bslib)
library(keras)


select <- dplyr::select
filter <- dplyr::filter
########################
roundUpNice <- function(x, nice=c(1,2,4,5,6,8,10)) {
  if(length(x) != 1) stop("'x' must be of length 1")
  10^floor(log10(x)) * nice[[which(x <= 10^floor(log10(x)) * nice)[[1]]]]
}

theme <- bs_theme(
  # Controls the default grayscale palette
  #bg = "#202123", fg = "#B8BCC2",
  bg = "white", fg = "#544F4E",
  bootswatch = 'simplex',
  # Controls the accent (e.g., hyperlink, button, etc) colors
  #primary = "#EA80FC", secondary = "#48DAC6",
  primary = "cadetblue", secondary = "#48DAC6",
  base_font = c("Grandstander", "sans-serif"),
  code_font = c("Courier", "monospace"),
  heading_font = "'Helvetica Neue', Helvetica, sans-serif",
  # Can also add lower-level customization
  "input-border-color" = "#EA80FC"
)

roundUpNice <- function(x, nice=c(1,2,4,5,6,8,10)) {
  if(length(x) != 1) stop("'x' must be of length 1")
  10^floor(log10(x)) * nice[[which(x <= 10^floor(log10(x)) * nice)[[1]]]]
}


yesterday <- format(Sys.Date()-1,"%d-%b")
today <- format(Sys.Date(),"%d-%b")

# Define UI for application that draws a histogram
ui <- fluidPage(theme = theme,
               fluidRow(column(titlePanel("SMART: A Statistical, Machine Learning Framework for Parametric Risk Transfer"),
                               width = 12,offset = 2)
               ),
               navbarPage("",
                          tabPanel(icon("home"),
                                   fluidRow(
                                     column(width = 3,
                                            imageOutput('logo_iuss', height = '30%'),
                                            br(),
                                            imageOutput('logo_reddom', height = '30%'),     
                                            br(),
                                            tags$img(src="https://upload.wikimedia.org/wikipedia/commons/2/27/Uni_Exeter.svg",width="100%%"),
                                     ),
                                     column(div(
                                       p("SMART is project funded by World Bank's Challenge Fund. An initiative of the Global Facility for Disaster Reduction and Recovery (GFDRR) and 
                                                      the UK’s Department for International Development (DFID) that has the purpose to bring innovation to developing countries afflicted by natural 
                                                      disasters like flood, hurricane and earthuake. \nIn this context, SMART tries to address these issues through the application of appropriate machine 
                                                      learning and statistical concepts to develop a new framework for parametric trigger modelling, using as a case study the Dominican Republic.
                                                      The project covers Thematic Area 2 of the Terms of Reference, entitled “Machine Learning and Big Data for Disaster Risk Financing.
                                                      Started in 2019 and reaching its conclsuion in June 2021, the project focus on two perils: flood and drought.\nThis web App is designed with the intent of showcasing the outcomes of the project but also to get the final users familiarise with machine learning
                                                        models and parametrics insurance."),
                                       HTML("<h3>Link to the repositories:</h3>"),
                                       p(HTML("<a href='https://github.com/luigicesarini/SMART'>Github Repository</a>")),
                                       HTML("<h3>Link to the final report of the project</h3>
                                            <a href='https://www.gfdrr.org/sites/default/files/SMART_Final_report_corrected.pdf'' target='_blank'>Final Report</a>"),
                                       style="text-align:justify;color:black;background-color:papayawhip;padding:15px;border-radius:10px"), 
                                       br(),
                                       fluidRow(
                                         column(6,
                                                br(),br(),br(),
                                                tags$img(src="https://landportal.org/sites/landportal.org/files/styles/220heightmax/public/GFDRR_WRC3.png?itok=Kt3ayAU9",width="100%%"),
                                         ),
                                         column(6,
                                                tags$img(src="https://upload.wikimedia.org/wikipedia/en/thumb/7/7c/DfID.svg/1200px-DfID.svg.png",width="100%"),)
                                       ),
                                       
                                       #tags$img(src="https://www.worldometers.info/img/maps/dominican_rep_physical_map.gif",width="75%"),
                                       width = 5),
                                     column(tags$img(src="https://www.worldometers.info/img/maps/dominican_rep_physical_map.gif",width="80%"),
                                            br(),br(),br(),
                                            tags$img(src="http://www.infomercatiesteri.it//public/images/paesi/131/files/world%20bank.jpg",width="60%", margin = 'auto'),
                                            width=4)
                                   )
                                   
                          ),
                          tabPanel("Identification of Flood Events",
                                   sidebarLayout(
                                     sidebarPanel(width=3,
                                       dateInput("dateIdentificationFlood",
                                                 "Select the day of interest:",
                                                 value = '2007-01-01',
                                                 min = '2003-01-04',
                                                 max = '2018-07-15',
                                                 #format = format('%d %b %Y'),
                                                 language = 'en',
                                                 width = '100%'
                                       ),
                                       div(
                                         HTML("<h3 style='text-align:center;'>Identification of flood events</h3>"),
                                         HTML("The biggest challenge in the application of parametric insurance as a risk reduction measure is presented by the basis risk, which represents the mismatch between the payouts and the occurrence of a damaging event (i.e., a flood event occurred but no payout was issued or viceversa).
                                             <br>	
                                            	Machine learning models can help towards a more objective and correct identification 
                                            	of these extreme events that would lead to the reduction of basis risk and to more prompt payouts.
                                             <br>
                                             	The methodology proposed uses open quasi-global gridded climate datasets to predict the occurrence of flood and drought
                                             	events. Support vector machines and neural networks were used to detect the events.
                                             <br>
                                             	The panel on the right, displays the rainfall from 4 different datasets on the date selected by the user, along with the prediction returned, in this web app, by the neural network.
                                             <br>
                                            	An article detailing information about the data, the methodology and presenting a comprehensive analysis and discussion of the results obtained can be found in the journal website , NHESS, at the following <a href='https://nhess.copernicus.org/preprints/nhess-2020-220/'>link</a>"),
                                         style="text-align:justify;color:black;background-color:'purple';padding:0px;border-radius:10px")),
                                     mainPanel(fluidRow(
                                       column(9,
                                              imageOutput("FloodMap")
                                       ),
                                       column(3, 
                                              htmlOutput('PredictionFlood'),
                                              br(),
                                              br(),
                                              tableOutput('PredictionFloodTable')
                                      ))))),
                          tabPanel("Identification of Drought Events",
                                   sidebarLayout(
                                     sidebarPanel(width=3,
                                       dateInput("dateIdentificationDrought",
                                                 "Select the date of interest:",
                                                 value = '2015-01-01',
                                                 min = '2003-06-11',
                                                 max = '2019-07-01',
                                                 #format = format('%d %b %Y'),
                                                 language = 'en',
                                                 width = '100%'
                                       ),
                                       div(
                                         HTML("<h3 style='text-align:center '>About drought predictions</h3>"),
                                         p(HTML("<p style='color:black;'>The prediction for the drought case are performed at weekly scale. Selecting a date form the menù,
                                                the corresponding week of the year, according to the ISO standard is retrieved and the corresponding prediction is returned.
                                                The maps illustrate the spatial distribution of the Standard Precipitation Index (SPI) over the Dominican Republic for the chosen week.</p>")),
                                         br(),
                                         p(HTML('The SPI is a widely adopted drought index, used as input to the machine learning algorithm. More details about the SPI and the methodology applied 
                                                can be found in the <a href="https://www.gfdrr.org/sites/default/files/SMART_Final_report_corrected.pdf" target="_blank">final report</a>.')),
                                         style="text-align:justify;color:black;background-color:'purple';padding:0px;border-radius:10px")),
                                     mainPanel(fluidRow(
                                       column(9,
                                              imageOutput("DroughtMap")
                                       ),
                                       column(3, 
                                              htmlOutput('PredictionDrought'),
                                              br(),
                                              br(),
                                              tableOutput('PredictionDroughtTable')
                                       ))))),
                          tabPanel("Dynamic Financial Analysis",
                                   sidebarLayout(
                                     sidebarPanel(
                                       sliderInput(inputId = "IntervalFlood",
                                                   label = "Period Insured in years",
                                                   min = 2003,
                                                   max = 2019,
                                                   value = c(2006,2010),
                                                   width = "220px",
                                                   sep = ''),

                                       HTML('<p>
                                             The quantification of the benefit provided by the implementation of a parametric insurance program is demonstrated 
                                             simulating the evolution, over a time window selected by the user, of a hypothetical disaster reserve funds and comparing is behaviour
                                             to the same reserve fund, but lacking any insurance.
                                             </p>
                                             <p>
                                             On the right is provided a dynamic financial analysis that simulates the reserve fund and the expenses consequent 
                                             to a series of events and payouts, computed according to the identification of the events provided
                                             by the machine learning algorithm.
                                             <br>
                                             In this illustrative example, payouts and losses are posed equal while the balances for the two scenarios are defined as follow:<br>
                                             <strong>Reserve fund without insurance: </strong>
                                             <br>Annual money allocated for natural disasters - Losses <br>
                                             <strong>Reserve fund WITH insurance: </strong>
                                             <br>Annual money allocated for natural disasters - Losses + Payouts - Premium insurance <br>
                                             
                                            </p>  
                                            <p>
                                              <span style="font-weight:1000;color:red;font-size:12px;">
                                                DISCLAIMER:
                                              </span>
                                              <br>
                                                The numbers provided in this platform are being used solely for illustrative purpose.
                                            </p> 
                                             '
                                            ),
                                       
                                     ),
                                     
                                     mainPanel(
                                       plotOutput("PayoutLosses",height = '300px',width = '100%'),
                                       br(),
                                       plotOutput("BalanceFlood",height = '300px',width = '100%')
                                     )
                                   )
                          ),
                          tabPanel("Milk Production Forecast",
                                   sidebarLayout(
                                     sidebarPanel(
                                       sliderInput('mon',
                                                   'Number of months to project',
                                                   min = 1,
                                                   max = 12,
                                                   value = 3,
                                                   step=1,
                                                   ticks=FALSE
                                       ),
                                       div(
                                       HTML('<h3 style="text-align:center">Milk Production forecast:</h3> 
                                        The dairy industry is a relevant sector in the Dominican Republic that is affected by natural disasters and
                                        extreme events, having an impact on the livelihood of people dependending on it. Predicting milk
                                        production in advance, could provide the improvements sought after towards the implementation of insurance product and/or 
                                        early warning system able to aid farmers in coping with the consequences of severe weather conditions.
                                        <br>
                                        Different deep learning models were tested to predict milk production. Here are reported the results, and forecasts,
                                        of the 1-dimensional convolutional neural network (CNN1D). The lack of data in the Dominincan Republic, open the door
                                        to the idea of <i>transfer learning</i> that is re-purpose a model trained on a task to perform the same task in another 
                                        context. In the case of the milk production for the Dominican Republic, we decided to train the deeep learning model
                                        in three European countries with longer record of data (480 months) and then using said model in the Dominican Republic. 
                                        The interactive plot show the comparison between the model trained in Europe and the one limited to the short Dominican 
                                        record (84 months).
                                        <br>
                                        Further information regarding the data, the methodology and a deeper analysis and discussion of the results can be found
                                        in the <a href="https://www.gfdrr.org/sites/default/files/SMART_Final_report_corrected.pdf" target="_blank">final report</a> of the project.'),
                                       style="text-align:justify;color:black;background-color:'purple';padding:0px;border-radius:10px") 
                                     ) ,
                                     mainPanel(
                                       plotlyOutput("milk_forecast",height = '600px')
                                     )
                                   )
                          )
                          
               )
               
)


# Define server fucntion
server <- function(input, output) {
  # ------------------ App virtualenv setup (Do not edit) ------------------- #
  if (Sys.info()[['user']]  == 'lcesarini') {
    reticulate::use_python(python = "/usr/local/bin/python3", required = TRUE)
  }else{
    virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
    python_path = Sys.getenv('PYTHON_PATH')
    # Create virtual env and install dependencies
    reticulate::virtualenv_create(envname = virtualenv_dir, python= python_path)
    reticulate::virtualenv_install(virtualenv_dir, packages = c('xarray', 'cartopy','matplotlib','numpy','pandas',"xarray[io]","xarray[complete]",
                                                                'netCDF4', 'h5netcdf', 'scipy', 'pydap', 'zarr', 'fsspec', 'cftime', 'rasterio', 'cfgrib', 'pooch'))
    reticulate::use_virtualenv(virtualenv_dir, required = TRUE)
    
  }
  print(Sys.info()[['user']])
  # reticulate::py_config()
  # reticulate::virtualenv_remove(envname = "python_environment")
  #   
  output$logo_iuss <- renderImage({
    return(list(
      src = "images/logo_iuss.png",
      contentType = "image/png",
      alt = "iuss", 
      width = '100%'
    ))
  }, deleteFile = FALSE)
  
  output$logo_exeter <- renderImage({
    list(src = "images/exeter_logo.png",
         width = "100%"
    )
  }, deleteFile = FALSE)
  
  output$logo_reddom <- renderImage({
    list(src = "images/reddom_logo.png",
         width = "100%"
    )
  }, deleteFile = FALSE)
  
  output$logo_wb <- renderImage({
    list(src = "images/logo_wb.jpeg",
         width = "100%"
    )
  }, deleteFile = FALSE)
  
  output$logo_cgdp <- renderImage({
    list(src = "images/cgdp.png",
         width = "100%"
    )
  }, deleteFile = FALSE)
  
  output$logo_gfdrr <- renderImage({
    list(src = "images/gfdrr.png",
         width = "100%"
    )
  }, deleteFile = FALSE)
  
  #=========#=========#=========#=========#=========#=========#=========#=========#
  #  Flood tab                                                     #
  #=========#=========#=========#=========#=========#=========#=========#=========#
  #bs_themer()
  df <- data.table::fread('df_predictions_app.csv')
  loss <- 100
  payout <- 100
  
  
  # PLOT 1 
  output$PayoutLosses <- renderPlot({
    
    df %>% 
      filter(year >= input$IntervalFlood[1],
             year <= input$IntervalFlood[2]) %>% 
      
      mutate(Loss   = loss*Output,
             Payout = -(payout*Prediction)
      ) -> df_filtered
    
    df_filtered %>% 
      mutate(Loss     = case_when(is.na(Loss) ~ 0, TRUE ~ Loss),
             Payout   = case_when(is.na(Payout) ~ 0, TRUE ~ Payout)) -> df_filtered
    
    cols_dynamic <- c('Loss' = 'red', 'Payout' = 'green')
    ggplot(df_filtered,aes(x=Date))+
      geom_segment(aes(xend =Date,
                       y    = ifelse(Loss == 0, NA,Loss),
                       yend = ifelse(Loss == 0, NA,0),
                       col = 'Loss'))+
      geom_segment(aes(xend =Date,
                       y    = ifelse(Payout == 0, NA,Payout),
                       yend = ifelse(Payout == 0, NA,0),
                       col = 'Payout'))+
      scale_color_manual(values = cols_dynamic)+
      labs(x = 'Years', y = 'Losses/Payouts in thousands of $', color = '',
           title = "Losses and payouts in case of flood",
           subtitle = glue::glue('AAL: {round(sum(abs(df_filtered$Payout))/(max(df_filtered$year)-min(df_filtered$year)),0)}K$'))+
      theme_bw()
    
    
  })
  
  
  
  output$BalanceFlood <- renderPlot({
    
    loan <- 200    
    
    df %>% 
      filter(year >= input$IntervalFlood[1],
             year <= input$IntervalFlood[2]) %>% 
      mutate(Loss     = loss*Output,
             Payout   = -(payout*Prediction),
             Loan     = case_when(lubridate::month(Date) == 1 & lubridate::day(Date) == 1 ~ loan,
                                  TRUE ~ 0)
      ) -> df_filtered
    
    premium <- round(sum(abs(df_filtered$Payout),na.rm=TRUE)/(max(df_filtered$year)-min(df_filtered$year)),0)*1.05
    
    df_filtered %>% 
      mutate(Loss     = case_when(is.na(Loss) ~ 0, TRUE ~ Loss),
             Payout   = case_when(is.na(Payout) ~ 0, TRUE ~ Payout),
             Premium  = case_when(lubridate::month(Date) == 1 & lubridate::day(Date) == 1 ~ premium,
                                  TRUE ~ 0),
             wo_insu  = -Loss+Loan,
             w_insur  = -Loss-Payout+Loan-Premium) -> df_filtered    
    
    cols_2 <-  c('Without Insurance'='black','Parametric Insurance'='blue')
    
    ggplot(df_filtered,aes(x=Date))+
      geom_line(aes(y = 5000+cumsum(-Loss+Loan), col = 'Without Insurance'), alpha = 0.7)+
      geom_line(aes(y = 5000+cumsum(-Loss-Payout+Loan-Premium), col = 'Parametric Insurance'))+
      geom_hline(aes(col = 'Without Insurance',yintercept = mean(5000+cumsum(wo_insu))), linetype = "dashed")+
      geom_hline(aes(col = 'Parametric Insurance',yintercept = mean(5000+cumsum(w_insur))), linetype = "dashed")+
      scale_color_manual(values = cols_2)+
      labs(x = 'Years', y = 'Reserve Funds balance in thousands of $',
           color = '',
           title = "Bank Account Flood Case",
           subtitle = glue::glue('AAL: {premium}K$'))+
      theme_bw()+
      theme(legend.position = 'right')
    
    
  })
  
  #=========#=========#=========#=========#=========#=========#=========#=========#
  #  Milk production tab                                                 #
  #=========#=========#=========#=========#=========#=========#=========#=========#
  

  output$milk_forecast <- renderPlotly({
    months_to_project <- ifelse(nchar(input$mon) == 1, paste0(0,input$mon),input$mon)
    
    prev_months <- fread('csv/milk_previous_months.csv', data.table = FALSE) %>% rename('Observed Milk'='milk','Date' = 'date')
    pred_months <- fread('csv/milk_all_country.csv', data.table = FALSE) %>% rename('Date'='V1', 'Observed Milk' = 'Observed')
    

    full_join(prev_months,pred_months, by = c('Date','Observed Milk')) %>% 
      reshape2::melt(id.vars = 'Date',value.name='Milk') %>% 
      mutate(Date = as.Date(Date)) %>% 
      filter(!is.na(Milk), Date <= as.Date(glue('2015-{months_to_project}-01')), Date >= as.Date('2014-10-01')) %>% 
      rename('Variable'='variable')-> df_plot 
      
    cols <- c('Observed Milk'='blue',
              'DOM'          = 'black',
              'Italy'        = 'red',
              'France'       = 'magenta',
              'Germany'      = 'darkgreen')
    
    ggplot(data = df_plot, aes(x=Date,y=Milk))+
      geom_line(aes(col = Variable))+
      geom_point(aes(col = Variable))+
      scale_color_manual(values=cols)+
      labs(y = 'Milk Production in mln of liters', x = '', colour = '',
           title = glue('{input$mon} months single step forecast for the Dominican milk production'),
           subtitle = glue::glue('{format(as.Date(glue("2015-01-01")), format = "%B-%Y")}/{format(as.Date(glue("2015-{months_to_project}-01")), format = "%B-%Y")}'))+
      theme_bw()+
      theme(legend.position = 'bottom')+
      coord_cartesian(ylim  = c(min(df_plot$Milk)-0.05*min(df_plot$Milk),
                                max(df_plot$Milk)+0.05*max(df_plot$Milk))) -> plot_milk
    
    
    plotly::ggplotly(plot_milk)
    
      
  })
  
  #=========#=========#=========#=========#=========#=========#=========#=========#
  #  Event identification Prec MAP                                                #
  #=========#=========#=========#=========#=========#=========#=========#=========#
  
  output$FloodMap <- renderImage({
    input$dateIdentificationFlood %>% write.table('csv/date_map_flood.txt', quote = FALSE,row.names = FALSE)
    
    if (is.null(input$dateIdentificationFlood))
      return(NULL)
    
    if (!is.null(input$dateIdentificationFlood)){
      
      reticulate::source_python('plot_maps_app.py')
      
      
      return(list(
        src = "images/flood_identification.jpg",
        contentType = "image/jpg",
        alt = "Face", 
        width = '100%'
      ))
      }
    
    
  }, deleteFile = FALSE)
  
  output$PredictionFlood <- renderText({
    
    occurrence <- ifelse( (fread('csv/prediction_flood.csv') %>% filter(Date == as.character(input$dateIdentificationFlood)) %>% pull(Prediction)) == 0,'DID NOT OCCURRED','OCCURRED')
    
    HTML(glue("<p style='font-weight:400;color:black;font-size:13pt;'>Model's Prediction:</p>A flood event <strong>{occurrence}</strong> on this date"))


  }) 
  
  output$PredictionFloodTable <- renderTable({
    
    fread('csv/prediction_flood.csv', data.table = FALSE) %>% 
      filter(Date == as.character(input$dateIdentificationFlood)) %>% 
      mutate(Date = as.character(Date)) %>% 
      t() %>% 
      data.frame() %>% 
      tibble::rownames_to_column() %>% 
      rename('Results'='.',
             'Summary'='rowname') %>% 
      gt::gt() 
    
  }
    
  )
  
  #=========#=========#=========#=========#=========#=========#=========#=========#
  #  Event identification SPI MAP                                                #
  #=========#=========#=========#=========#=========#=========#=========#=========#
  
  output$DroughtMap <- renderImage({
    
    year <- as.Date(as.character(input$dateIdentificationDrought)) %>% lubridate::year()
    week <- as.Date(as.character(input$dateIdentificationDrought)) %>% lubridate::week()
    
    paste(year,week,sep = '_') %>% 
      write.table('csv/date_map_drought.txt', quote = FALSE,row.names = FALSE)
    
    if (is.null(input$dateIdentificationDrought))
      return(NULL)
    
    if (!is.null(input$dateIdentificationDrought)){
      
      reticulate::source_python('plot_maps_app_drought.py')
      
      
      return(list(
        src = "images/drought_identification.jpg",
        contentType = "image/jpg",
        alt = "Face", 
        width = '100%'
      ))
    }
    
    
  }, deleteFile = FALSE)
  
  output$PredictionDrought <- renderText({
    
    year <- as.Date(as.character(input$dateIdentificationDrought)) %>% lubridate::year()
    week <- as.Date(as.character(input$dateIdentificationDrought)) %>% lubridate::week()
    
    occurrence <- ifelse( (fread('csv/prediction_drought.csv') %>% filter(Year == year, Week == week) %>% pull(Prediction)) == 0,'are NOT','are')
    
    HTML(glue("Model's Prediction:\nWe {} in drought condition"))
    
    HTML(glue("<p style='font-weight:400;color:black;font-size:13pt;'>Model's Prediction:</p>We <strong>{occurrence}</strong> in drought condition"))
    
    
  }) 
  
  output$PredictionDroughtTable <- renderTable({
    
    year <- as.Date(as.character(input$dateIdentificationDrought)) %>% lubridate::year()
    week <- as.Date(as.character(input$dateIdentificationDrought)) %>% lubridate::week()
    
    
    fread('csv/prediction_drought.csv', data.table = FALSE) %>%
      filter(Year == year,
             Week == week) %>%
      t() %>%
      data.frame() %>%
      tibble::rownames_to_column() %>%
      rename('Results'='.',
             'Summary'='rowname') %>%
      gt::gt()

  }
  
  )
}

# Run the application 
shinyApp(ui = ui, server = server)
