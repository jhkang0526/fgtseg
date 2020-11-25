def file_read(fpt):
    
    IMG = 'img'
    
    subject_dict = {
        IMG: fpt,
    }
    print('Dataset size:', len(subject_dict), 'subjects')

    return subject_dict