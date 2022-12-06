#pragma once

#define template_generator(CLASS_NAME, TYPE_NAME) \
    template class CLASS_NAME<TYPE_NAME>;

#define template_generation(CLASS_NAME)                                            \
    template_generator(CLASS_NAME, short)                                          \
        template_generator(CLASS_NAME, int)                                        \
            template_generator(CLASS_NAME, long)                                   \
                template_generator(CLASS_NAME, long long)                          \
                    template_generator(CLASS_NAME, unsigned short)                 \
                        template_generator(CLASS_NAME, unsigned int)               \
                            template_generator(CLASS_NAME, unsigned long)          \
                                template_generator(CLASS_NAME, unsigned long long) \
                                    template_generator(CLASS_NAME, float)          \
                                        template_generator(CLASS_NAME, double)
