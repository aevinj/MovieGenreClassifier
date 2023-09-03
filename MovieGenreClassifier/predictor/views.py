from django.shortcuts import render
from django.http import HttpResponse
from .main import MovieGenreClassifier

def predictor(request):
    if request.method == 'POST':
        mgc = MovieGenreClassifier()
        custom_plot_option = request.POST.get('custom_plot_option')
        
        if custom_plot_option == 'on':
            result = mgc.predict_genre(request.POST.get('user_input'))
        else:
            result = mgc.predict_genre()
        return render(request, 'result.html', {'result': result})
    else:
        return HttpResponse("something went wrong!")