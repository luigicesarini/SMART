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
library(reticulate)

#use_python(python = "/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7", required = TRUE)
select <- dplyr::select
filter <- dplyr::filter
########################
roundUpNice <- function(x, nice=c(1,2,4,5,6,8,10)) {
  if(length(x) != 1) stop("'x' must be of length 1")
  10^floor(log10(x)) * nice[[which(x <= 10^floor(log10(x)) * nice)[[1]]]]
}
# if (RCurl::url.exists("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv")) {
#     df <- fread('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv') %>% 
#         filter(!is.na(as.numeric(totale_casi)))
# }else{
#     print("The URL does not exist. Check the repository")
# }
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
ui <- fillPage(theme = theme,
               fluidRow(column(titlePanel("SMART: A Statistical, Machine Learning Framework for Parametric Risk Transfer"),
                               width = 12,offset = 2)
                        ),
                navbarPage("",
                           tabPanel(icon("home"),
                                    fluidRow(
                                             column(width = 3,
                                                    tags$img(src="https://lh3.googleusercontent.com/proxy/lb1oYgo5WURSwO1gHyoklIwXpjabdc6Qn9Lhc0vojSqFmHkTqP1DDF68oqX1DutosXDDpjusJboFzg9RyXE46Q7ER9kphDjhljVvn_8p1FzI9Wqsig",width="100%%"),
                                                    br(),
                                                    tags$img(src="https://lh3.googleusercontent.com/proxy/0gOp55zEr2_np6MMdr-_d82bMvlzC2AGvGVxSONvUiDhfEjdE5g33oto1pyvIUzSFguHiYDpI6_v5lS7uWpBQ_j8sY892UUIwK1ZObrqIuMkEEIKjHe_OSqI0_e73zGKjQ",width="100%%"),
                                                    br(),br(),br(),br(),br(),
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
                                                      p("Link to the repositories:"),
                                                      p(HTML("<a href='https://github.com/luigicesarini/SMART'>Github Repository</a>")),
                                                      style="text-align:justify;color:black;background-color:papayawhip;padding:15px;border-radius:10px"), 
                                                    br(),
                                                    fluidRow(
                                                      column(6,
                                                             br(),br(),br(),
                                                             tags$img(src="https://landportal.org/sites/landportal.org/files/styles/220heightmax/public/GFDRR_WRC3.png?itok=Kt3ayAU9",width="100%%"),
                                                             ),
                                                      column(6,
                                                             tags$img(src="http://www.infomercatiesteri.it//public/images/paesi/131/files/world%20bank.jpg",width="100%"),)
                                                    ),
                                                    
                                                    #tags$img(src="https://www.worldometers.info/img/maps/dominican_rep_physical_map.gif",width="75%"),
                                                    width = 5),
                                             column(tags$img(src="https://www.worldometers.info/img/maps/dominican_rep_physical_map.gif",width="100%"),
                                                    br(),br(),br(),
                                                    tags$img(src="https://upload.wikimedia.org/wikipedia/en/thumb/7/7c/DfID.svg/1200px-DfID.svg.png",width="50%%"),
                                                    width=4)
                                             )
                             
                           ),
                           tabPanel("Identification of flood events",
                                    sidebarLayout(
                                      sidebarPanel(
                                        dateInput("dateIdentification",
                                                 "Select the day of interest:",
                                                 value = '2007-01-01',
                                                 min = '2003-01-04',
                                                 max = '2018-07-15',
                                                 #format = format('%d %b %Y'),
                                                 language = 'en',
                                                 width = '100%'
                                        ),
                                        br(),
                                        br(),
                                        br(),
                                        div(
                                        h3("Model's Prediction:"),
                                        p(HTML("<h2 style='color:red;'>A flood event OCCURRED on this date</h2>")),
                                        style="text-align:center;color:black;background-color:'purple';padding:15px;border-radius:10px")),
                                      mainPanel(
                                        plotOutput("prec_map",height = '500px')
                                      )
                                    )
                           ),
                           tabPanel("Flood",
                                    sidebarLayout(
                                      sidebarPanel(
                                        sliderInput(inputId = "IntervalFlood",
                                                    label = "Period Insured in years",
                                                    min = 2003,
                                                    max = 2019,
                                                    value = c(2006,2010),
                                                    width = "220px",
                                                    sep = ''),
                                        radioButtons("TypeCrop",
                                                     "Crops:",
                                                     selected = "Banana",
                                                     choiceNames =
                                                       list('Banana',
                                                            'Coffee',
                                                            'Rice'),
                                                     choiceValues =
                                                       list("Banana",
                                                            "Coffee",
                                                            "Rice")),
                                        helpText(HTML('<br><br><br><br><br>
                                                      <p style="font-weight:1000;color:red;font-size:16px;">DISCLAIMER:</p> 
                                                      The number provided in this platform are being used solely for illustrative purpose..\nbla bla bla write whatever we want in here')),
                                        
                                      ),
                                      
                                      mainPanel(
                                        fluidRow(
                                          column(width = 8,
                                                 plotOutput("PayoutLosses",height = '300px',width = '100%')
                                                 ),
                                          column(width = 4,
                                                 plotlyOutput('PieCrops',height = '300px', width ='100%')
                                                 )
                                        ),
                                        br(),
                                        plotOutput("BalanceFlood",height = '300px',width = '75%')
                                      )
                                    )
                           ),
                           tabPanel("Drought",
                                    sidebarLayout(
                                      sidebarPanel(
                                        sliderInput(inputId = "IntervalDrought",
                                                    label = "Period Insured in years",
                                                    min = 2003,
                                                    max = 2019,
                                                    value = c(2006,2010),
                                                    width = "220px",
                                                    sep = ''),
                                        radioButtons("TypeCropDR",
                                                     "Crops:",
                                                     selected = "Banana",
                                                     choiceNames =
                                                       list('Banana',
                                                            'Coffee',
                                                            'Rice'),
                                                     choiceValues =
                                                       list("Banana",
                                                            "Coffee",
                                                            "Rice")),
                                        helpText(HTML('<br><br><br><br><br>
                                                      <p style="font-weight:1000;color:red;font-size:16px;">DISCLAIMER:</p> 
                                                      The number provided in this platform are being used solely for illustrative purpose..\nbla bla bla write whatever we want in here')),
                                        
                                      ),
                                      
                                      mainPanel(
                                        fluidRow(
                                          column(width = 8,
                                                 plotOutput("PayoutLossesDR",height = '300px',width = '100%')
                                          ),
                                          column(width = 4,
                                                 plotlyOutput('PieCropsDR',height = '300px', width ='100%')
                                          )
                                        ),
                                        br(),
                                        plotOutput("BalanceFloodDR",height = '300px',width = '75%')
                                      )
                                    )
                           ),
                           tabPanel("Milk Production Forecast",
                                    sidebarLayout(
                                      sidebarPanel(
                                        sliderInput('mon',
                                                    '# Months to project',
                                                    min = 1,
                                                    max = 12,
                                                    value = 6,
                                                    step=1,
                                                    ticks=FALSE
                                        ),
                                        helpText(HTML('<p style="font-weight:1000;color:red;font-size:16px;">Province tab:</p> 
                                        Plot 1) Daily increase of positive case. 2) Total number of positive cases.'))
                                        
                                      ),
                                      mainPanel(
                                        plotOutput("milk_forecast",height = '600px')
                                      )
                                    )
                           )

                )

)

# Define server fucntion
server <- function(input, output) {
    output$logo_iuss <- renderImage({
      list(src = "images/logo_iuss.png",
      width = "100%"
      )
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

      if(input$TypeCrop == 'Banana'){
        crop_par <- 0.1378961
      }else if(input$TypeCrop == 'Coffee'){
        crop_par <- 0.3723768
      }else{
        crop_par <- 0.4897272
      }
      df %>% 
        filter(year >= input$IntervalFlood[1],
               year <= input$IntervalFlood[2]) %>% 
        
        mutate(Loss   = loss*Output,
               Payout = -(payout*Prediction)
        ) -> df_filtered
     
      df_filtered %>% 
        mutate(Loss     = case_when(is.na(Loss) ~ 0, TRUE ~ Loss),
               Payout   = case_when(is.na(Payout) ~ 0, TRUE ~ Payout)) -> df_filtered
      
      ggplot(df_filtered,aes(x=Date))+
        geom_segment(aes(xend =Date,
                         y    = ifelse(Loss == 0, NA,Loss*crop_par),
                         yend = ifelse(Loss == 0, NA,0)), col = 'red')+
        geom_segment(aes(xend =Date,
                         y    = ifelse(Payout == 0, NA,Payout*crop_par),
                         yend = ifelse(Payout == 0, NA,0)), col = 'green')+
        
        labs(x = 'Years', y = 'Losses in thousands of $',
             title = "Losses and payouts in case of flood",
             subtitle = glue::glue('AAL: {round(sum(abs(df_filtered$Payout))*crop_par/(max(df_filtered$year)-min(df_filtered$year)),0)}K$'))+
        theme_bw()
      
      
    })
    
    output$PieCrops <- renderPlotly({
      
      library(plotly)
      
      dom_crops <- data.frame("Crops"=c('Banana','Coffee','Rice'), 'Area' = c(55138.5,148896.9,195820.1))
      
      fig <- plot_ly(dom_crops, labels = ~Crops, values = ~Area, type = 'pie',textinfo='label+percent')
      fig <- fig %>% layout(title = list(text='Area harvested by crop\nDominican Republic',
                                         y = 0.9),
                            #xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                            #yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                            legend = list(y=0.5))
      
      fig
    })
    
    output$BalanceFlood <- renderPlot({
      if(input$TypeCrop == 'Banana'){
        crop_par <- 0.1378961
        loan <- 149
      }else if(input$TypeCrop == 'Coffee'){
        crop_par <- 0.3723768
        loan <- 200
      }else{
        crop_par <- 0.4897272
        loan <- 231
      }
      
      df %>% 
        filter(year >= input$IntervalFlood[1],
               year <= input$IntervalFlood[2]) %>% 
        mutate(Loss     = loss*Output*crop_par,
               Payout   = -(payout*Prediction*crop_par),
               Loan     = case_when(lubridate::month(Date) == 1 & lubridate::day(Date) == 1 ~ loan*crop_par,
                                    TRUE ~ 0)
        ) -> df_filtered
      
      premium <- round(sum(abs(df_filtered$Payout),na.rm=TRUE)/(max(df_filtered$year)-min(df_filtered$year)),0)
      
      df_filtered %>% 
        mutate(Loss     = case_when(is.na(Loss) ~ 0, TRUE ~ Loss),
               Payout   = case_when(is.na(Payout) ~ 0, TRUE ~ Payout),
               Premium  = case_when(lubridate::month(Date) == 1 & lubridate::day(Date) == 1 ~ premium,
                                    TRUE ~ 0),
               wo_insu  = -Loss+Loan,
               w_insur  = -Loss-Payout+Loan-Premium) -> df_filtered    
      
      cols_2 <-  c('Without Insurance'='black','Parametric Insurance'='blue')
      
      ggplot(df_filtered,aes(x=Date))+
        geom_line(aes(y = cumsum(-Loss+Loan), col = 'Without Insurance'), alpha = 0.7)+
        geom_line(aes(y = cumsum(-Loss-Payout+Loan-Premium), col = 'Parametric Insurance'))+
        geom_hline(aes(col = 'Without Insurance',yintercept = mean(cumsum(wo_insu))), linetype = "dashed")+
        geom_hline(aes(col = 'Parametric Insurance',yintercept = mean(cumsum(w_insur))), linetype = "dashed")+
        scale_color_manual(values = cols_2)+
        labs(x = 'Years', y = 'Losses in thousands of $',
             color = '',
             title = "Bank Account Flood Case",
             subtitle = glue::glue('AAL: {premium}K$'))+
        theme_bw()+
        theme(legend.position = 'bottom')
      
      
    })
    
    #=========#=========#=========#=========#=========#=========#=========#=========#
    #  Milk production tab                                                 #
    #=========#=========#=========#=========#=========#=========#=========#=========#
    
    year_to_forecast <- sample(2009:2019,1)
    
     output$milk_forecast <- renderPlot({
       library(dplyr)
      months_to_project <- ifelse(nchar(input$mon) == 1, paste0(0,input$mon),input$mon)
      data  <- data.table::fread("entire_milk_ts.txt", data.table = FALSE)
        mean_milk <- data %>% filter(as.Date(date) >= as.Date("1981-01-01"), as.Date(date) < as.Date("2009-01-01")) %>% pull(milk) %>% mean()
        sd_milk   <- data %>% filter(as.Date(date) >= as.Date("1981-01-01"), as.Date(date) < as.Date("2009-01-01")) %>% pull(milk) %>% sd()
        model <- keras::load_model_tf(filepath = "M4_MS_EVs_France_W13_FH12/")
        
        #year_to_forecast <- sample(2009:2019,1)
        
        
        input_milk <- data %>% filter(as.Date(date) >= as.Date(glue("{year_to_forecast-2}-12-01")),
                                 as.Date(date) < as.Date(glue("{year_to_forecast}-01-01"))) %>% dplyr::select(milk)
        output_milk <- data %>% filter(as.Date(date) >= as.Date(glue("{year_to_forecast-1}-06-01")),
                                  as.Date(date) < as.Date(glue("{year_to_forecast+1}-01-01"))) %>% dplyr::select(milk)

        inp_array <- array((input_milk$milk-mean_milk)/sd_milk, dim = c(1,13,1))
        model %>% predict(inp_array) -> predictions

        cols <- c('Ground Truth' = 'black', 'Forecast' = 'green')
        ggplot(data = data.frame('date'=seq.Date(as.Date(glue("{year_to_forecast-1}-06-01")), by = "month", length.out = 7+input$mon),
                                 'milk'=output_milk[1:(7+input$mon),'milk']),
               aes(x = date,y =milk))+
            geom_line(aes(col = 'Ground Truth'))+
            geom_point(aes(col = 'Ground Truth'))+
            geom_line(data = cbind('date_2'=seq.Date(as.Date(glue("{year_to_forecast}-01-01")), by = "month", length.out = input$mon),
                                   data.frame('milk'=t(predictions*sd_milk+mean_milk)[1:input$mon])),
                      aes(x = date_2,y =milk,col = 'Forecast'),
            )+
            geom_point(data = cbind('date_2'=seq.Date(as.Date(glue("{year_to_forecast}-01-01")), by = "month", length.out = input$mon),
                                    data.frame('milk'=t(predictions*sd_milk+mean_milk)[1:input$mon])),
                       aes(x = date_2,y =milk, group = seq_along(date_2),col = 'Forecast'),
            )+
            scale_color_manual(values=cols)+
            labs(y = 'Milk Production in mln of liters', x = '', colour = '',
                 title = glue('{input$mon} months forecast for France milk production'),
                 subtitle = glue::glue('{format(as.Date(glue("{year_to_forecast}-01-01")), format = "%B-%Y")}/{format(as.Date(glue("{year_to_forecast}-{months_to_project}-01")), format = "%B-%Y")}'))+
            theme_bw()+
            theme(legend.position = 'bottom')+
            coord_cartesian(ylim  = c(min(output_milk,t(predictions*sd_milk+mean_milk))-0.05*min(output_milk,t(predictions*sd_milk+mean_milk)),
                                      max(output_milk,t(predictions*sd_milk+mean_milk))+0.05*max(output_milk,t(predictions*sd_milk+mean_milk))))
    })

     #=========#=========#=========#=========#=========#=========#=========#=========#
     #  Event identification Prec MAP                                                #
     #=========#=========#=========#=========#=========#=========#=========#=========#
     
    output$prec_map <- renderPlot({
      DOM <- st_read('spatial/gadm36_DOM.gpkg', layer = 'gadm36_DOM_0')
      Country=c("DOM")
      Rainfall_DS=c("CCS","IMERG","PERSIANN","GSMAP","CHIRPS","CMORPH", "SM_ERA5")
      Rainfall_DS_Start=c("2003-01-01","2000-06-01","2000-03-01","2000-03-01","1981-01-01","1998-01-01","1981-01-01")
      list_DS <- vector(mode="list",6)
      rainfallds <- list()
      coords_list <- list()
      rainfall_melt <- list()
      z=0
      #RAINFALL ----
      for(i in 1:6){
        z=z+i/i
        
        filename <- list.files(paste0("csv/",Rainfall_DS[i], "/CSV/"),".csv")
        csvRainFile <- paste0("csv/",Rainfall_DS[i], "/CSV/",filename[grep(Country,filename)])
        rainfall <- fread(csvRainFile, header = TRUE)
        
        name_coords <- list.files(paste0("csv/",Rainfall_DS[i], "/RainfallCoords/"),".csv",full.names = TRUE)
        coords_File <- paste0(name_coords[grep(Country,name_coords)])
        coords <- fread(coords_File)
        
        
        #rainfallds[[Rainfall_DS[i]]] <- rainfall
        coords_list[[Rainfall_DS[i]]] <- coords
        rainfall %>%
          select('2003-04-17') %>% 
          cbind(coords[,c("X","Y")],.) %>% 
          melt(id.vars = c("X","Y")) -> rainfall_melt[[Rainfall_DS[i]]]
        
      }  
      
      par(mfrow=c(2,2))
      for (i in c(1,3,4,5)) {
        rainfall_melt[[Rainfall_DS[i]]] %>% 
          dplyr::select(-variable) %>% 
          as.data.frame() %>% 
          raster::rasterFromXYZ() %>% 
          raster::plot(col = tmaptools::get_brewer_pal('Blues', plot= FALSE),
                       main = glue::glue('{Rainfall_DS[i]}'))
          plot(DOM$geom,add=TRUE)
      }
    
    }
    )
        
        
        
        
}

# Run the application 
shinyApp(ui = ui, server = server)
