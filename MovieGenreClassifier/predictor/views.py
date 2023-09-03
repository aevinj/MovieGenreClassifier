from django.shortcuts import render
from django.http import HttpResponse

def predictor(request):
    return HttpResponse("This is ur prediction lol.")

def index(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        # Load your pre-trained model and make predictions here
        #predicted_genre = your_model.predict(user_input)
        #return render(request, 'result.html', {'predicted_genre': predicted_genre})
        print("hi")

    #return render(request, 'index.html')
    return HttpResponse("hi")