shinyServer(function(input, output, session) {
  output$myChart <- renderVis({
    with(tweetvis,
         createJSON(
           phi = tweetvis$phi, 
           theta = tweetvis$theta, 
           doc.length = tweetvis$doc.length, 
           vocab = tweetvis$vocab, 
           term.frequency = tweetvis$term.frequency,
           R = input$nTerms))})
})