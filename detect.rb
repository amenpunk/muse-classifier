require "pp"
require "rmagick"
require "json"
# Put nagadomi/animeface-2009 path here
require "/home/user/Development/animeface-2009/animeface-ruby/AnimeFace.so"

if ARGV.size == 0
    warn "Usage: #{$0} <source-file>"
    exit(-1)
end

image = Magick::ImageList.new(ARGV[0])
faces = AnimeFace::detect(image)
puts JSON.generate(faces)
