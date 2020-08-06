from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse

from core.utils import extract_images, run_image_comparison, remove_files


def upload_file(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        image_counter = extract_images(uploaded_file, "imgs_collection", store_to_db=True)
        context.update({'result': f"Successfully saved {image_counter} images"})
    return render(request, 'upload_file.html', context)


def home(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        remove_files('core/static/imgs/input_images')
        image_counter = extract_images(uploaded_file, "input_images")
        print(f'Found {image_counter}')
        return HttpResponseRedirect(reverse('results'))
    return render(request, 'plagiarism_test.html')


def results(request):
    found_results = run_image_comparison()
    context = {'results': found_results}
    return render(request, 'test_result.html', context=context)
